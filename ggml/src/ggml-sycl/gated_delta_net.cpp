#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>


template <int S_v, bool KDA, bool keep_rs_t>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Subgroup-16 GDN prefill specialization adapted from the engine's gdn_xe recurrence.
//
// The engine kernel assumes its own (row, head, column) state layout. GGML stores
// GDN state as [row, column, head, sequence] with row contiguous, so we retain
// GGML's coalesced row mapping and borrow the useful part of the Xe kernel:
// stage each Q/K vector once per 16-column workgroup, double-buffer it in local
// memory, and prefetch token t+1 while computing token t. Each subgroup handles
// four output columns in registers; four subgroups cover 16 columns per workgroup.
template <int S_v>
static void launch_gated_delta_net_sg16_prefill(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta,
        const float * curr_state, float * dst, float * state,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t n_k_heads, const sycl::uint3 rq3_magic,
        float scale, int64_t state_slot_stride, int K,
        dpct::queue_ptr stream) {
    static_assert(S_v == 128, "the SG16 GDN prefill tile is specialized for state width 128");
    constexpr int SG = 16;
    constexpr int LANES_PER_WG = 64;
    constexpr int COLS_PER_SG = 4;
    constexpr int COLS_PER_WG = 16;
    constexpr int ROWS_PER_LANE = S_v / SG;
    constexpr int COL_BLOCKS = S_v / COLS_PER_WG;

    const size_t n_groups = (size_t) n_seqs * (size_t) H * COL_BLOCKS;
    stream->submit([&](sycl::handler & cgh) {
        // Two banks, each containing one Q and one K vector (2*S floats).
        sycl::local_accessor<float, 1> qk_banks(sycl::range<1>(4 * S_v), cgh);
        const sycl::ext::oneapi::experimental::properties props{
            sycl::ext::intel::experimental::grf_size<256>};
        cgh.parallel_for<class GgmlGdnSg16Prefill>(
            sycl::nd_range<1>(sycl::range<1>(n_groups * LANES_PER_WG),
                              sycl::range<1>(LANES_PER_WG)),
            props,
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const auto sg = it.get_sub_group();
                const unsigned lane = (unsigned) sg.get_local_linear_id();
                const unsigned sg_id = (unsigned) sg.get_group_linear_id();
                const unsigned tid = (unsigned) it.get_local_linear_id();

                const size_t group_id = it.get_group_linear_id();
                const unsigned cb = (unsigned) (group_id % COL_BLOCKS);
                const size_t head_seq = group_id / COL_BLOCKS;
                const int64_t head = (int64_t) (head_seq % (size_t) H);
                const int64_t seq = (int64_t) (head_seq / (size_t) H);
                const uint32_t q_head = (uint32_t) (head % n_k_heads);
                const uint32_t q_seq = fastdiv((uint32_t) seq, rq3_magic);

                const int col[COLS_PER_SG] = {
                    (int) (cb * COLS_PER_WG + sg_id * COLS_PER_SG + 0),
                    (int) (cb * COLS_PER_WG + sg_id * COLS_PER_SG + 1),
                    (int) (cb * COLS_PER_WG + sg_id * COLS_PER_SG + 2),
                    (int) (cb * COLS_PER_WG + sg_id * COLS_PER_SG + 3),
                };

                const float * state_in[COLS_PER_SG];
                int64_t state_base[COLS_PER_SG];
                float s[COLS_PER_SG][ROWS_PER_LANE];
#pragma unroll
                for (int c = 0; c < COLS_PER_SG; ++c) {
                    // GGML state layout: [row, column, head, sequence], row contiguous.
                    // Keep the per-column base so every requested rollback snapshot can
                    // be written with the same sequence/head/column mapping.
                    const int64_t base = ((seq * H + head) * S_v + col[c]) * S_v;
                    state_base[c] = base;
                    state_in[c] = curr_state + base;
#pragma unroll
                    for (int r = 0; r < ROWS_PER_LANE; ++r) {
                        s[c][r] = state_in[c][r * SG + lane];
                    }
                }

                float nq[2], nk[2], nv[COLS_PER_SG], ng = 0.0f, nb = 0.0f;
                auto fetch = [&](int64_t t) {
                    const float * q_t = q + (int64_t) q_seq * sq3 + t * sq2 + (int64_t) q_head * sq1;
                    const float * k_t = k + (int64_t) q_seq * sq3 + t * sq2 + (int64_t) q_head * sq1;
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        nq[u] = q_t[tid + u * LANES_PER_WG];
                        nk[u] = k_t[tid + u * LANES_PER_WG];
                    }
                    const float * v_t = v + seq * sv3 + t * sv2 + head * sv1;
#pragma unroll
                    for (int c = 0; c < COLS_PER_SG; ++c) nv[c] = v_t[col[c]];
                    const int64_t gb = seq * sb3 + t * sb2 + head * sb1;
                    ng = g[gb];
                    nb = beta[gb];
                };

                if (n_tokens > 0) fetch(0);
                for (int64_t t = 0; t < n_tokens; ++t) {
                    const unsigned bank = (unsigned) (t & 1) * (2 * S_v);
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        qk_banks[bank + tid + u * LANES_PER_WG] = nq[u];
                        qk_banks[bank + S_v + tid + u * LANES_PER_WG] = nk[u];
                    }
                    const float v_now[COLS_PER_SG] = { nv[0], nv[1], nv[2], nv[3] };
                    const float gate_now = ng;
                    const float beta_now = nb;

                    it.barrier(sycl::access::fence_space::local_space);
                    // Global loads for the next token are issued before this token's recurrence.
                    if (t + 1 < n_tokens) fetch(t + 1);

                    const float decay = sycl::native::exp(gate_now);
#pragma unroll
                    for (int c = 0; c < COLS_PER_SG; ++c) {
                        float kv_partial = 0.0f;
#pragma unroll
                        for (int r = 0; r < ROWS_PER_LANE; ++r) {
                            const int i = r * SG + (int) lane;
                            kv_partial += s[c][r] * qk_banks[bank + S_v + i];
                        }
                        // A subgroup handles one output column at a time; all 16 lanes
                        // therefore contribute rows for the same column.
                        const float kv = warp_reduce_sum<SG>(kv_partial);
                        const float delta = (v_now[c] - decay * kv) * beta_now;

                        float out_partial = 0.0f;
#pragma unroll
                        for (int r = 0; r < ROWS_PER_LANE; ++r) {
                            const int i = r * SG + (int) lane;
                            s[c][r] = decay * s[c][r] + qk_banks[bank + S_v + i] * delta;
                            out_partial += s[c][r] * qk_banks[bank + i];
                        }
                        const float out_col = warp_reduce_sum<SG>(out_partial) * scale;
                        if (lane == 0) {
                            dst[((seq * n_tokens + t) * H + head) * S_v + col[c]] = out_col;
                        }
                    }

                    // GGML rollback convention: slot 0 is newest, slot 1 is one
                    // token older.  For a chunk longer than K, only the last K
                    // post-token states are retained.  This matches the reference
                    // GDN implementation for both fused-cache and ordinary outputs.
                    const int64_t target_slot = n_tokens - 1 - t;
                    if (target_slot >= 0 && target_slot < K) {
#pragma unroll
                        for (int c = 0; c < COLS_PER_SG; ++c) {
                            float * snapshot = state + target_slot * state_slot_stride + state_base[c];
#pragma unroll
                            for (int r = 0; r < ROWS_PER_LANE; ++r) {
                                snapshot[r * SG + lane] = s[c][r];
                            }
                        }
                    }
                }
            });
    });
}

static bool ggml_sycl_gdn_sg16_prefill_requested() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_SYCL_GDN_SG16_PREFILL");
        // Default on for devices that pass the SG16 capability and shape checks.
        // Set GGML_SYCL_GDN_SG16_PREFILL=0 to force the established recurrence.
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

static bool ggml_sycl_gdn_sg16_device(dpct::queue_ptr stream) {
    const auto device = stream->get_device();
    if (!device.is_gpu() || device.get_info<sycl::info::device::vendor_id>() != 0x8086) {
        return false;
    }
    const auto subgroup_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    return std::find(subgroup_sizes.begin(), subgroup_sizes.end(), size_t(16)) != subgroup_sizes.end() &&
           device.get_info<sycl::info::device::max_work_group_size>() >= 64 &&
           device.get_info<sycl::info::device::local_mem_size>() >= 4 * 128 * sizeof(float);
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    const int num_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            {
                constexpr int sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            break;
        case 32:
            {
                constexpr int sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                     });
            }
            break;
        case 64: {
            {
                constexpr int sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            break;
        }
        case 128: {
            {
                constexpr int sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                                        });
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;
    const sycl::uint3 rq3_magic = init_fastdiv_values(rq3);

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // Experimental SG16 GDN prefill path.  It supports ordinary GDN and the
    // K>1 rollback-snapshot output used by fused-cache graphs.  Keep T>=2 so
    // single-token decode continues to use the established recurrence kernel.
    const int64_t state_slot_min_stride = S_v * S_v * H * n_seqs;
    const bool sg16_shape_supported = !kda && S_v == 128 && n_tokens >= 2 &&
                                      K >= 1 && neqk1 > 0 && H % neqk1 == 0 &&
                                      (K == 1 || state_slot_stride >= state_slot_min_stride);
    if (ggml_sycl_gdn_sg16_prefill_requested() && sg16_shape_supported) {
        dpct::queue_ptr gdn_stream = ctx.stream();
        if (ggml_sycl_gdn_sg16_device(gdn_stream)) {
            GGML_SYCL_DEBUG("[SYCL] GDN SG16 prefill: Q/K double-buffered, 16-column workgroups (S=%lld, H_v=%lld, H_k=%lld, K=%d, T=%lld, seq=%lld)\n",
                            (long long) S_v, (long long) H, (long long) neqk1, K,
                            (long long) n_tokens, (long long) n_seqs);
            launch_gated_delta_net_sg16_prefill<128>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1, rq3_magic, scale, state_slot_stride, K, gdn_stream);
            return;
        }
        GGML_SYCL_DEBUG("[SYCL] GDN SG16 prefill: device unsupported, using existing recurrence\n");
    } else if (ggml_sycl_gdn_sg16_prefill_requested()) {
        GGML_SYCL_DEBUG("[SYCL] GDN SG16 prefill: shape/op unsupported (S=%lld, K=%d, KDA=%d, H=%lld, H_k=%lld, T=%lld, state_slot_stride=%lld), using existing recurrence\n",
                        (long long) S_v, K, (int) kda, (long long) H,
                        (long long) neqk1, (long long) n_tokens, (long long) state_slot_stride);
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
