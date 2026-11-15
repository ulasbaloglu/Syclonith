#include "llama-moe-cache-sycl.h"

#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

static std::vector<ggml_tensor *> llama_moe_sycl_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> result;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            result.push_back(t);
        }
    }
    return result;
}

static ggml_backend_buffer_t llama_moe_sycl_root_buffer(const ggml_tensor * tensor) {
    const ggml_tensor * root = tensor;
    while (root->view_src != nullptr) {
        root = root->view_src;
    }
    return root->buffer;
}

static bool llama_moe_sycl_is_host_weight(const ggml_tensor * tensor) {
    const ggml_backend_buffer_t buffer = llama_moe_sycl_root_buffer(tensor);
    return tensor->data != nullptr && buffer != nullptr &&
        ggml_backend_buffer_is_host(buffer) &&
        ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_n_dims(tensor) == 3 && ggml_is_contiguous(tensor);
}

static bool llama_moe_sycl_same_layout(const std::vector<ggml_tensor *> & experts) {
    if (experts.empty()) {
        return false;
    }
    for (const ggml_tensor * tensor : experts) {
        if (!llama_moe_sycl_is_host_weight(tensor) || tensor->ne[2] != experts[0]->ne[2]) {
            return false;
        }
    }
    return true;
}

static ggml_context_ptr make_context(size_t n_tensors) {
    ggml_init_params params = {
        /*.mem_size   =*/ std::max<size_t>(1, n_tensors) * ggml_tensor_overhead() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to create SYCL MoE cache tensor context");
    }
    return ctx;
}

} // namespace

struct llama_moe_cache_sycl::impl {
    struct layer_cache {
        int32_t il = -1;
        ggml_backend_t backend = nullptr;
        std::vector<ggml_tensor *> experts;
        std::vector<ggml_tensor *> banks;
        ggml_tensor * slot_map = nullptr;
        std::vector<int32_t> slot_to_expert;
        std::vector<uint64_t> last_used;
        uint64_t clock = 0;
        int32_t n_slots = 0;
        int32_t n_expert = 0;
        size_t bytes_per_slot = 0;
        ggml_context_ptr bank_ctx;
        ggml_backend_buffer_ptr bank_buffer;
    };

    ggml_context_ptr host_ctx;
    ggml_backend_buffer_ptr host_buffer;
    std::vector<layer_cache> layers;
    std::unordered_map<int32_t, size_t> layer_index;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> weight_banks;
    size_t budget_bytes = 0;
    size_t allocated_bytes = 0;
    size_t n_hits = 0;
    size_t n_misses = 0;
    size_t n_uploads = 0;
    size_t uploaded_bytes = 0;

    ~impl() {
        if (!layers.empty()) {
            LLAMA_LOG_INFO("%s: hits=%zu misses=%zu uploads=%zu uploaded=%.1f MiB\n",
                __func__, n_hits, n_misses, n_uploads, uploaded_bytes / 1048576.0);
        }
    }

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, size_t requested_budget) {
        if (requested_budget == 0 || model.hparams.n_expert == 0 || model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            return;
        }

        struct candidate {
            int32_t il;
            ggml_backend_t backend;
            std::vector<ggml_tensor *> experts;
            size_t bytes_per_slot;
        };
        std::vector<candidate> candidates;
        size_t total_host_bytes = 0;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const ggml_backend_dev_t layer_dev = model.dev_layer((int) il);
            if (layer_dev == nullptr) {
                continue;
            }
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(layer_dev);
            const char * reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
            if (reg_name == nullptr || std::strcmp(reg_name, "SYCL") != 0) {
                continue;
            }
            auto backend_it = std::find_if(backends.begin(), backends.end(), [&](ggml_backend_t backend) {
                return backend != nullptr && ggml_backend_get_device(backend) == layer_dev;
            });
            if (backend_it == backends.end()) {
                continue;
            }
            const auto dev_type = ggml_backend_dev_type(layer_dev);
            if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }

            std::vector<ggml_tensor *> experts = llama_moe_sycl_layer_experts(model.layers[il]);
            if (!llama_moe_sycl_same_layout(experts)) {
                continue;
            }

            size_t bytes_per_slot = 0;
            bool compatible = true;
            for (const ggml_tensor * tensor : experts) {
                if (tensor->nb[2] == 0 || tensor->nb[2] > SIZE_MAX - bytes_per_slot) {
                    compatible = false;
                    break;
                }
                bytes_per_slot += tensor->nb[2];
            }
            if (!compatible || bytes_per_slot == 0 ||
                bytes_per_slot > SIZE_MAX / (size_t) experts[0]->ne[2]) {
                continue;
            }
            const size_t layer_bytes = bytes_per_slot * (size_t) experts[0]->ne[2];
            if (layer_bytes > SIZE_MAX - total_host_bytes) {
                throw std::runtime_error("SYCL MoE cache size overflow");
            }
            total_host_bytes += layer_bytes;
            candidates.push_back({ (int32_t) il, *backend_it, std::move(experts), bytes_per_slot });
        }

        if (candidates.empty() || total_host_bytes == 0) {
            return;
        }

        // The requested budget is a hard cap. Do not silently grow it to the
        // device's reported free-memory figure.
        budget_bytes = requested_budget;
        const int32_t min_slots = std::max<int32_t>(1, (int32_t) model.hparams.n_expert_used_max());

        for (const candidate & c : candidates) {
            if (c.experts[0]->ne[2] > INT32_MAX) {
                continue;
            }
            const int32_t n_expert = (int32_t) c.experts[0]->ne[2];
            const size_t layer_bytes = c.bytes_per_slot * (size_t) n_expert;
            const size_t layer_budget = (size_t) ((long double) budget_bytes * (long double) layer_bytes / (long double) total_host_bytes);
            int32_t n_slots = (int32_t) std::min<size_t>((size_t) n_expert, layer_budget / c.bytes_per_slot);
            if (n_slots < min_slots) {
                continue;
            }
            layer_cache l;
            l.il = c.il;
            l.backend = c.backend;
            l.experts = c.experts;
            l.n_expert = n_expert;
            l.n_slots = n_slots;
            l.bytes_per_slot = c.bytes_per_slot;
            l.slot_to_expert.assign(n_slots, -1);
            l.last_used.assign(n_slots, 0);
            layers.push_back(std::move(l));
        }

        if (layers.empty()) {
            budget_bytes = 0;
            return;
        }

        // Allocate host slot maps in one ordinary CPU buffer. They are small
        // and are copied to the SYCL device by the scheduler callback.
        host_ctx = make_context(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            layer_cache & l = layers[i];
            l.slot_map = ggml_new_tensor_2d(host_ctx.get(), GGML_TYPE_I32, 1, l.n_expert);
            ggml_format_name(l.slot_map, "sycl_moe_cache.slot_map.%d", l.il);
            layer_index.emplace(l.il, i);
        }
        host_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.get(), ggml_backend_cpu_buffer_type()));
        if (!host_buffer) {
            throw std::runtime_error("failed to allocate SYCL MoE cache slot maps");
        }
        ggml_backend_buffer_clear(host_buffer.get(), 0xff); // every expert starts uncached
        // Mark slot maps as host weights. The scheduler then starts a new split
        // at their GET_ROWS consumer, after routed IDs have been produced.
        ggml_backend_buffer_set_usage(host_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (layer_cache & l : layers) {
            l.bank_ctx = make_context(l.experts.size());
            for (const ggml_tensor * expert : l.experts) {
                ggml_tensor * bank = ggml_new_tensor_3d(l.bank_ctx.get(), expert->type,
                    expert->ne[0], expert->ne[1], l.n_slots);
                if (bank->nb[2] != expert->nb[2]) {
                    throw std::runtime_error("SYCL MoE cache bank stride differs from source expert stride");
                }
                ggml_format_name(bank, "sycl_moe_cache.bank.%d.%s", l.il, expert->name);
                l.banks.push_back(bank);
                weight_banks.emplace(expert, bank);
            }
            const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(l.backend);
            l.bank_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(l.bank_ctx.get(), buft));
            if (!l.bank_buffer) {
                throw std::runtime_error("failed to allocate SYCL MoE cache expert bank");
            }
            ggml_backend_buffer_clear(l.bank_buffer.get(), 0);
            ggml_backend_buffer_set_usage(l.bank_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            allocated_bytes += (size_t) l.n_slots * l.bytes_per_slot;

            // If the entire expert set fits, make it resident once at initialization.
            // Partial caches remain empty and are filled on demand by the copy callback.
            if (l.n_slots == l.n_expert) {
                int32_t * map = (int32_t *) l.slot_map->data;
                for (int32_t e = 0; e < l.n_expert; ++e) {
                    map[e] = e;
                    l.slot_to_expert[e] = e;
                    l.last_used[e] = ++l.clock;
                }
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * expert = l.experts[ip];
                    ggml_backend_tensor_set_async(l.backend, l.banks[ip], expert->data, 0, ggml_nbytes(l.banks[ip]));
                }
                ggml_backend_synchronize(l.backend);
            }
        }

        LLAMA_LOG_INFO("%s: enabled for %zu SYCL MoE layers, %.1f MiB allocated (budget %.1f MiB); CPU miss fallback is not enabled in Stage A\n",
            __func__, layers.size(), allocated_bytes / 1048576.0, budget_bytes / 1048576.0);
    }

    layer_cache * find_layer(int32_t il) {
        const auto it = layer_index.find(il);
        return it == layer_index.end() ? nullptr : &layers[it->second];
    }

    const layer_cache * find_layer(int32_t il) const {
        const auto it = layer_index.find(il);
        return it == layer_index.end() ? nullptr : &layers[it->second];
    }
};

llama_moe_cache_sycl::llama_moe_cache_sycl(const llama_model & model,
        const std::vector<ggml_backend_t> & backends, size_t budget_bytes) :
        pimpl(new impl(model, backends, budget_bytes)) {
}

llama_moe_cache_sycl::~llama_moe_cache_sycl() = default;

bool llama_moe_cache_sycl::active() const {
    return pimpl && !pimpl->layers.empty();
}

ggml_backend_t llama_moe_cache_sycl::backend(int32_t il) const {
    const impl::layer_cache * l = pimpl ? pimpl->find_layer(il) : nullptr;
    return l ? l->backend : nullptr;
}

ggml_tensor * llama_moe_cache_sycl::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    const impl::layer_cache * l = pimpl ? pimpl->find_layer(il) : nullptr;
    if (l == nullptr || n_tokens <= 0 || n_expert_used <= 0) {
        return nullptr;
    }
    if (l->n_slots != l->n_expert &&
        (n_tokens > 32 || n_tokens > INT64_MAX / n_expert_used || n_tokens * n_expert_used > l->n_slots)) {
        return nullptr;
    }
    return l->slot_map;
}

ggml_tensor * llama_moe_cache_sycl::get_experts(const ggml_tensor * weight) const {
    if (!pimpl || weight == nullptr) {
        return nullptr;
    }
    const auto it = pimpl->weight_banks.find(weight);
    return it == pimpl->weight_banks.end() ? nullptr : it->second;
}

bool llama_moe_cache_sycl::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    if (!pimpl || src == nullptr || dst == nullptr || graph == nullptr) {
        return false;
    }
    impl::layer_cache * layer = nullptr;
    for (auto & candidate : pimpl->layers) {
        if (candidate.slot_map == src) {
            layer = &candidate;
            break;
        }
    }
    if (layer == nullptr) {
        return false;
    }
    if (backend != layer->backend) {
        GGML_ABORT("SYCL MoE cache layer %d reached backend %s instead of its assigned backend %s",
            layer->il, ggml_backend_name(backend), ggml_backend_name(layer->backend));
    }
    GGML_ASSERT(ggml_nbytes(dst) == ggml_nbytes(layer->slot_map));

    if (layer->n_slots == layer->n_expert) {
        const int32_t * map = (const int32_t *) layer->slot_map->data;
        ggml_backend_tensor_set_async(backend, dst, map, 0, ggml_nbytes(dst));
        return true;
    }

    const ggml_tensor * lookup = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
            lookup = node;
            break;
        }
    }
    if (lookup == nullptr || lookup->src[1] == nullptr) {
        GGML_ABORT("SYCL MoE cache could not find the slot-map GET_ROWS node for layer %d", layer->il);
    }

    const ggml_tensor * ids = lookup->src[1];
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == ids || node == ids->view_src) {
            GGML_ABORT("SYCL MoE cache layer %d: routed expert IDs must be computed in a preceding scheduler split", layer->il);
        }
    }
    if (ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) || ggml_nelements(ids) > (int64_t) layer->n_slots) {
        GGML_ABORT("SYCL MoE cache layer %d: unsupported routed-ID tensor shape", layer->il);
    }

    std::vector<int32_t> selected((size_t) ggml_nelements(ids));
    ggml_backend_tensor_get(ids, selected.data(), 0, ggml_nbytes(ids));

    int32_t * slot_map = (int32_t *) layer->slot_map->data;
    std::vector<int32_t> unique;
    unique.reserve(selected.size());
    for (int32_t expert : selected) {
        if (expert == -1) {
            continue;
        }
        if (expert < 0 || expert >= layer->n_expert) {
            GGML_ABORT("SYCL MoE cache layer %d: routed expert ID %d is out of range", layer->il, expert);
        }
        if (std::find(unique.begin(), unique.end(), expert) == unique.end()) {
            unique.push_back(expert);
        }
    }
    if (unique.size() > (size_t) layer->n_slots) {
        GGML_ABORT("SYCL MoE cache layer %d: selected %zu unique experts for %d slots", layer->il, unique.size(), layer->n_slots);
    }

    // Keep hits resident and refresh their LRU timestamps before admitting misses.
    size_t layer_hits = 0;
    size_t layer_misses = 0;
    for (int32_t expert : unique) {
        const int32_t slot = slot_map[expert];
        if (slot >= 0) {
            layer->last_used[slot] = ++layer->clock;
            layer_hits++;
        } else {
            layer_misses++;
        }
    }

    for (int32_t expert : unique) {
        if (slot_map[expert] >= 0) {
            continue;
        }
        int32_t slot = -1;
        for (int32_t i = 0; i < layer->n_slots; ++i) {
            if (layer->slot_to_expert[i] < 0) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            slot = (int32_t) (std::min_element(layer->last_used.begin(), layer->last_used.end()) - layer->last_used.begin());
            const int32_t evicted = layer->slot_to_expert[slot];
            if (evicted >= 0) {
                slot_map[evicted] = -1;
            }
        }
        for (size_t ip = 0; ip < layer->experts.size(); ++ip) {
            const ggml_tensor * expert_tensor = layer->experts[ip];
            ggml_backend_tensor_set_async(backend, layer->banks[ip],
                (const uint8_t *) expert_tensor->data + (size_t) expert * expert_tensor->nb[2],
                (size_t) slot * layer->banks[ip]->nb[2], expert_tensor->nb[2]);
        }
        slot_map[expert] = slot;
        layer->slot_to_expert[slot] = expert;
        layer->last_used[slot] = ++layer->clock;
        pimpl->n_uploads++;
        pimpl->uploaded_bytes += layer->bytes_per_slot;
    }
    pimpl->n_hits += layer_hits;
    pimpl->n_misses += layer_misses;

    // Enqueue after expert uploads on the same backend stream: the following
    // GET_ROWS and MUL_MAT_ID cannot observe a slot before its bytes arrive.
    ggml_backend_tensor_set_async(backend, dst, slot_map, 0, ggml_nbytes(dst));
    return true;
}

bool llama_moe_cache_sycl::copy_callback(ggml_backend_t backend, const ggml_tensor * src,
        ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * cache = static_cast<llama_moe_cache_sycl *>(user_data);
    return cache != nullptr && cache->copy(backend, src, dst, graph);
}
