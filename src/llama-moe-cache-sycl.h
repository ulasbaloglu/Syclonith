#pragma once

#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;

// A conservative, SYCL-only expert-slot cache. It deliberately does not implement
// CPU miss execution, prefetch, or CUDA-specific cache features.
class llama_moe_cache_sycl {
public:
    llama_moe_cache_sycl(const llama_model & model,
                         const std::vector<ggml_backend_t> & backends,
                         size_t budget_bytes);
    ~llama_moe_cache_sycl();

    llama_moe_cache_sycl(const llama_moe_cache_sycl &) = delete;
    llama_moe_cache_sycl & operator=(const llama_moe_cache_sycl &) = delete;

    bool active() const;
    ggml_backend_t backend(int32_t il) const;
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;
    ggml_tensor * get_experts(const ggml_tensor * weight) const;

    // Called by the backend scheduler when it copies a slot-map input to a
    // SYCL backend. It reads the already-computed routed IDs, fills missing
    // expert slots, then uploads the updated slot map.
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    static bool copy_callback(ggml_backend_t backend, const ggml_tensor * src,
                              ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
