#include "fused-gemm.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <tuple>
#include <type_traits>
#include <mutex>
#include <set>
#include <unordered_map>

namespace mx = sycl::ext::oneapi::experimental::matrix;

// FG_ / fg_ is short for fused GEMM: the weights are dequantized inside the GEMM, into the XMX tiles.

// A k step is one 32-value weight sub-block; iq3_s and the other superblock formats split their
// superblock into steps of this width. The sub-groups of a work-group each walk their own K range
// and are summed at the end.
static constexpr int FG_BK     = QK4_NL;
static constexpr int FG_KSPLIT = 4;

// Element traits of one joint_matrix operand type. The A stage and the B pack compute in f32 and
// convert once, in registers, when they write the element, so any type costs the same one pass.
//   store: storage in SLM (A) and in the packed B buffer
//   mtype: matrix_type in matrix_combinations
//   mode:  GGML_SYCL_DYNAMIC_PRECISION value that selects this type
//   src:   ggml type that needs no conversion into this type (GGML_TYPE_COUNT: none)
//   slow:  XMX throughput class, 0 is fastest. f16 and bf16 share the DPAS rate; tf32 does half the
//          K per instruction. B60, Qwen3-30B-A3B pp512: f16 1108, bf16 1000, tf32 751 t/s
template <typename T> struct fg_elem;

template <> struct fg_elem<sycl::half> {
    using store = sycl::half;
    using pair  = sycl::half2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::fp16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_F16;
    static constexpr ggml_type       src   = GGML_TYPE_F16;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return (store) x; }
    static pair make(float x, float y) { return pair((store) x, (store) y); }
};

// tf32 rounds to nearest even with plain bit ops: round_to_tf32 needs a SPIR-V extension that the
// DG2 AOT target rejects
static inline uint32_t fg_round_bits(float x, int drop) {
    const uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        return (u & 0x7fffffu) ? u | (1u << drop) : u; // nan stays nan
    }
    return u + ((1u << (drop - 1)) - 1) + ((u >> drop) & 1);
}

struct alignas(4) fg_bf16x2 {
    sycl::ext::oneapi::bfloat16 x, y;
};

template <> struct fg_elem<sycl::ext::oneapi::bfloat16> {
    using store = sycl::ext::oneapi::bfloat16;
    using pair  = fg_bf16x2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::bf16;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_BF16;
    static constexpr ggml_type       src   = GGML_TYPE_BF16;
    static constexpr int             mant  = 7;
    static constexpr int             slow  = 0;
    static store cvt(float x) { return store(x); }
    static pair make(float x, float y) { return { cvt(x), cvt(y) }; }
};

// tf32 keeps f32 range and f16 mantissa, in f32 storage
template <> struct fg_elem<mx::precision::tf32> {
    using store = float;
    using pair  = sycl::float2;
    static constexpr mx::matrix_type mtype = mx::matrix_type::tf32;
    static constexpr int             mode  = GGML_SYCL_DYNAMIC_PRECISION_TF32;
    static constexpr ggml_type       src   = GGML_TYPE_COUNT;
    static constexpr int             mant  = 10;
    static constexpr int             slow  = 1;
    static store cvt(float x) { return sycl::bit_cast<float>(fg_round_bits(x, 13) & ~0x1fffu); }
    static pair make(float x, float y) { return pair(cvt(x), cvt(y)); }
};

// One joint_matrix combination (A type, B type, TM x TN x TK, sub-group size; C and D are f32) and
// the tiling built on it. A sub-group owns SG_ROWS rows of A (at least 16) and BN (at least 32)
// columns of B. A and B may differ: the device lists the pairs it supports.
template <typename TA, typename TB, int TM_, int TN_, int TK_, int SG_> struct fg_combo {
    using ta  = TA;
    using tb  = TB;
    using EA  = fg_elem<TA>;
    using EB  = fg_elem<TB>;
    using tsa = typename EA::store;
    using tsb = typename EB::store;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int TK = TK_;
    static constexpr int SG = SG_;
    static constexpr int VNNI    = 4 / sizeof(tsb);  // K rows of B packed in one 32-bit word
    static constexpr int SG_ROWS = TM > 16 ? TM : 16;
    static constexpr int RPL     = SG_ROWS / SG;     // A rows one lane decodes per k step
    static constexpr int MT      = SG_ROWS / TM;
    static constexpr int BN      = TN > 32 ? TN : 32;
    static constexpr int NT      = BN / TN;
    static constexpr int WG_SIZE = FG_KSPLIT * SG;
    static constexpr mx::layout b_layout = VNNI == 1 ? mx::layout::row_major : mx::layout::ext_intel_packed;
    // a 64-wide N is mostly padding here and a 32x64 f32 accumulator needs 128 registers per lane,
    // so it spills: 13x slower on B60
    static constexpr bool efficient = TN <= 32;
    static_assert(SG_ROWS % SG == 0 && SG_ROWS % TM == 0 && BN % TN == 0 && FG_BK % TK == 0, "bad tile");
    static_assert(BN <= GGML_SYCL_FG_MAX_N, "header gate must cover the tile width");
};

using fg_half = sycl::half;
using fg_bf16 = sycl::ext::oneapi::bfloat16;
using fg_tf32 = mx::precision::tf32;

// One bit of GGML_SYCL_XMX_GATHER_SHAPES per combination. Only combinations some device lists in
// matrix_combinations are built (appendix of sycl_ext_oneapi_matrix and the runtime's own list).
template <typename F> static void fg_visit_combo(int idx, F && f);
static constexpr int FG_N_COMBOS = 8;

// A spir64_gen AOT build (GGML_SYCL_XMX_AOT_SG) drops the combinations of the other sub-group size
// entirely: ocloc rejects even an empty kernel that asks for a sub-group size it lacks.
template <int SG> static constexpr bool fg_listed() {
#if defined(GGML_SYCL_XMX_AOT_SG)
    return SG == GGML_SYCL_XMX_AOT_SG;
#else
    return true;
#endif
}

template <typename S, typename F> static void fg_call_combo(F && f) {
    if constexpr (fg_listed<S::SG>()) {
        f(S{});
    }
}

template <typename F> static void fg_visit_combo(int idx, F && f) {
    switch (idx) {
        case 0: fg_call_combo<fg_combo<fg_half, fg_half, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 1: fg_call_combo<fg_combo<fg_half, fg_half, 16, 16, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 2: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 16, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 3: fg_call_combo<fg_combo<fg_half, fg_half, 32, 64, 32, 16>>(f); break; // Xe2, Xe3, Xe-HPC
        case 4: fg_call_combo<fg_combo<fg_half, fg_half, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        case 5: fg_call_combo<fg_combo<fg_tf32, fg_tf32, 8, 16, 8, 16>>(f);   break; // Xe2, Xe3, Xe-HPC
        case 6: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 16, 16, 16>>(f);  break; // Xe2, Xe3, Xe-HPC
        case 7: fg_call_combo<fg_combo<fg_bf16, fg_bf16, 8, 8, 16, 8>>(f);    break; // Xe-HPG (Arc A), ARL-H
        default: GGML_ABORT("bad XMX combination %d", idx);
    }
}

// AOT with -fsycl-targets=intel_gpu_*: compile each tile body only for targets with its sub-group
// size, since IGC fails on the other ones. A JIT build keeps them all, but each combination lands in
// its own device image (joint_matrix is an optional kernel feature) and only a combination the
// device reports is launched, so the runtime never asks IGC for the others.
#if defined(__SYCL_DEVICE_ONLY__)
#    if __SYCL_TARGET_INTEL_GPU_ACM_G10__ || __SYCL_TARGET_INTEL_GPU_ACM_G11__ || __SYCL_TARGET_INTEL_GPU_ACM_G12__ || \
        __SYCL_TARGET_INTEL_GPU_ARL_H__
#        define FG_AOT_SG 8
#    elif __SYCL_TARGET_INTEL_GPU_PVC__ || __SYCL_TARGET_INTEL_GPU_PVC_VG__ || __SYCL_TARGET_INTEL_GPU_BMG_G21__ || \
        __SYCL_TARGET_INTEL_GPU_BMG_G31__ || __SYCL_TARGET_INTEL_GPU_LNL_M__ || __SYCL_TARGET_INTEL_GPU_PTL_H__ ||   \
        __SYCL_TARGET_INTEL_GPU_PTL_U__ || __SYCL_TARGET_INTEL_GPU_WCL__ || __SYCL_TARGET_INTEL_GPU_NVL_S__ ||       \
        __SYCL_TARGET_INTEL_GPU_NVL_U__ || __SYCL_TARGET_INTEL_GPU_NVL_P__
#        define FG_AOT_SG 16
#    elif __SYCL_TARGET_INTEL_GPU_TGLLP__ || __SYCL_TARGET_INTEL_GPU_RKL__ || __SYCL_TARGET_INTEL_GPU_ADL_S__ || \
        __SYCL_TARGET_INTEL_GPU_ADL_P__ || __SYCL_TARGET_INTEL_GPU_ADL_N__ || __SYCL_TARGET_INTEL_GPU_DG1__ ||   \
        __SYCL_TARGET_INTEL_GPU_MTL_U__ || __SYCL_TARGET_INTEL_GPU_MTL_H__
#        define FG_AOT_SG 0 // no XMX
#    endif
#endif

template <int SG> static constexpr bool fg_built() {
#if defined(FG_AOT_SG)
    return SG == FG_AOT_SG;
#else
    return true;
#endif
}

// Upper bound on the tile count when total_rows rows are routed to n_as experts: the worst case gives
// each expert one row and fills whole tiles with the rest. The bound depends only on the shape, not
// on the routing, so the pool reuses one buffer every ubatch instead of keeping one per size seen.
static constexpr int64_t grouped_gemm_max_tiles(int64_t total_rows, int64_t n_as, int64_t BN) {
    return total_rows <= n_as ? total_rows : n_as + (total_rows - n_as) / BN;
}
// Tiles do not cross experts, so the bound is not ceil(total_rows / BN): 34 rows over 2 experts with
// BN = 16 split 17 + 17 need 2 + 2 tiles, where the ceil gives 3.
static_assert(grouped_gemm_max_tiles(34, 2, 16) == 4);

// the device lists S with an f32 accumulator and output
template <typename S> static bool fg_device_has_combo(const std::vector<mx::combination> & combinations) {
    for (const auto & c : combinations) {
        if (c.atype == S::EA::mtype && c.btype == S::EB::mtype && c.ctype == mx::matrix_type::fp32 &&
            c.dtype == mx::matrix_type::fp32 &&
            (c.max_msize >= (size_t) S::TM || c.msize == (size_t) S::TM) &&
            (c.max_nsize >= (size_t) S::TN || c.nsize == (size_t) S::TN) &&
            (c.max_ksize >= (size_t) S::TK || c.ksize == (size_t) S::TK)) {
            return true;
        }
    }
    return false;
}

template <typename T> static const char * fg_type_name() {
    return std::is_same_v<T, fg_half> ? "f16" : std::is_same_v<T, fg_bf16> ? "bf16" : "tf32";
}

static std::string fg_combo_name(int idx) {
    std::string name;
    fg_visit_combo(idx, [&](auto s) {
        using S = decltype(s);
        name = std::string(fg_type_name<typename S::ta>()) + "x" + fg_type_name<typename S::tb>() + " " +
               std::to_string(S::TM) + "x" + std::to_string(S::TN) + "x" + std::to_string(S::TK) + " sg" +
               std::to_string(S::SG);
    });
    return name;
}

// Combinations this build has kernels for and the device lists, one bit each. Cached per device:
// on a mixed box the first caller's verdict is not the others'.
static int fg_device_combos(const sycl::device & dev) {
    static std::mutex                            mtx;
    static std::unordered_map<sycl::device, int> known;
    std::lock_guard<std::mutex>                  lock(mtx);
    const auto                                   it = known.find(dev);
    if (it != known.end()) {
        return it->second;
    }
    int available = 0;
    try {
        const auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
        const auto sg_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                const bool sg_ok = std::find(sg_sizes.begin(), sg_sizes.end(), (size_t) S::SG) != sg_sizes.end();
                if (sg_ok && fg_device_has_combo<S>(combinations)) {
                    available |= 1 << idx;
                }
            });
        }
    } catch (const sycl::exception &) {
        available = 0;
    }
    GGML_LOG_INFO("%s: %s: XMX dequant-GEMM combinations available 0x%x, allowed 0x%x\n", __func__,
                  dev.get_info<sycl::info::device::name>().c_str(), available, g_ggml_sycl_xmx_gather_shapes);
    known.emplace(dev, available);
    return available;
}

// Rank of combination S for a src1 of type src1_type, lower is better. Order:
//  1. throughput: a tile that does not spill, then the fastest type class of A and B
//  2. B type equal to the src1 type, so the pack is a plain copy
//  3. B at least as precise as f16
//  4. the device's native DPAS tile (8 x SG x 32 bytes of K), then the largest M x K
// A costs nothing to convert: the A stage emits any type at the same cost.
template <typename S> static int64_t fg_rank(ggml_type src1_type) {
    const int64_t spills  = !S::efficient;
    const int64_t slow    = std::max(S::EA::slow, S::EB::slow);
    const int64_t convert = S::EB::src != src1_type;
    const int64_t lossy   = S::EB::mant < 10;
    const int64_t foreign = !(S::TM == 8 && S::TN == S::SG);
    const int64_t mk      = 1024 - S::TM * S::TK;
    return ((((spills * 2 + slow) * 2 + convert) * 2 + lossy) * 2 + foreign) * 2048 + mk;
}

// whether XMX operands of type mode (a GGML_SYCL_DYNAMIC_PRECISION value) meet the src1 request
// [TAG_GGML_PREC]. f16 lacks the f32 range that BF16 and F32 ask for; an F32 request goes only as far
// down as GGML_SYCL_DYNAMIC_REQUIRED_PRECISION allows.
static bool fg_mode_meets(int mode, int32_t src1_prec) {
    if (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16) {
        return true;
    }
    if (src1_prec >= GGML_PREC_BF16) {
        return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
    }
    switch (g_ggml_sycl_dynamic_required_precision) {
        case GGML_SYCL_DYNAMIC_PRECISION_TF32: return mode == GGML_SYCL_DYNAMIC_PRECISION_TF32;
        case GGML_SYCL_DYNAMIC_PRECISION_BF16: return mode != GGML_SYCL_DYNAMIC_PRECISION_F16;
        default:                               return false;
    }
}

// Best allowed combination for this call, or -1 if none. The type is GGML_SYCL_DYNAMIC_PRECISION if it
// meets the src1 request; if not, bf16 then tf32 for a BF16 request (fastest first) and tf32 then bf16
// for an F32 request (most mantissa first).
static int fg_pick_combo(dpct::queue_ptr stream, ggml_type src1_type, int32_t src1_prec) {
    const sycl::device dev     = stream->get_device();
    const int          allowed = fg_device_combos(dev) & g_ggml_sycl_xmx_gather_shapes;
    const bool         f32_req = src1_prec != GGML_PREC_UNDEFINED && src1_prec < GGML_PREC_BF16;
    const int          modes[] = {
        g_ggml_sycl_dynamic_precision,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_TF32 : GGML_SYCL_DYNAMIC_PRECISION_BF16,
        f32_req ? GGML_SYCL_DYNAMIC_PRECISION_BF16 : GGML_SYCL_DYNAMIC_PRECISION_TF32,
    };
    int     best      = -1;
    int64_t best_rank = 0;
    for (int i = 0; i < 3 && best < 0; ++i) {
        const int mode = modes[i];
        if (!fg_mode_meets(mode, src1_prec)) {
            continue;
        }
        for (int idx = 0; idx < FG_N_COMBOS; ++idx) {
            if (!(allowed & (1 << idx))) {
                continue;
            }
            fg_visit_combo(idx, [&](auto s) {
                using S = decltype(s);
                if (S::EA::mode != mode || S::EB::mode != mode) {
                    return;
                }
                const int64_t rank = fg_rank<S>(src1_type);
                if (best < 0 || rank < best_rank) {
                    best      = idx;
                    best_rank = rank;
                }
            });
        }
    }
    // log each distinct decision once
    static std::mutex                                      mtx;
    static std::set<std::tuple<size_t, int, int32_t, int>> seen;
    std::lock_guard<std::mutex>                            lock(mtx);
    if (seen.emplace(std::hash<sycl::device>{}(dev), (int) src1_type, src1_prec, best).second) {
        GGML_LOG_INFO("%s: src1 %s, src1 prec %d -> %s\n", __func__, ggml_type_name(src1_type), src1_prec,
                      best >= 0 ? fg_combo_name(best).c_str() : "none (library GEMM)");
    }
    return best;
}

// src1 [N][K] -> packed [K/V][Npad][V] so B tiles load straight from global memory
template <typename E, typename T_src>
static void fused_gemm_pack_b(const T_src * y, typename E::store * packed, int N, int Npad, int K,
                              dpct::queue_ptr stream) {
    constexpr int V   = 4 / sizeof(typename E::store);
    const int     kqs = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const int idx = id[0];
        const int n   = idx / kqs;
        const int kq  = idx - n * kqs;
        typename E::store vals[V] = {};
        if (n < N) {
            const T_src * src = y + (size_t) n * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt((float) src[v]);
            }
        }
        typename E::store * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// A stage: one lane owns one row and decodes FG_BK values of it per k step, with every scale
// folded into the value so the mad below sees plain A elements. One overload per weight format.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_nl * __restrict__ xrow, const int kb, typename E::pair * a) {
    const block_iq4_nl blk = xrow[kb];
    const float        d   = (float) blk.d;
#pragma unroll
    for (int j = 0; j < QK4_NL / 2; j += 2) {
        const uint8_t q0 = blk.qs[j];
        const uint8_t q1 = blk.qs[j + 1];
        a[j / 2]     = E::make(d * kvalues_iq4nl[q0 & 0xf], d * kvalues_iq4nl[q1 & 0xf]);
        a[j / 2 + 8] = E::make(d * kvalues_iq4nl[q0 >> 4], d * kvalues_iq4nl[q1 >> 4]);
    }
}

// iq3_s: k step kb is sub-block kb % 8 of superblock kb / 8. The superblock is 110 bytes, so read
// only the fields of that sub-block instead of copying the block. Same decode as
// dequantize_block_iq3_s: grid entries are taken as dwords and the sign bit is a plain shift.
template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    static_assert(QK_K == 256, "the iq3_s A stage assumes 8 sub-blocks per superblock");
    const block_iq3_s * blk = xrow + kb / (QK_K / 32);
    const int           ib8 = kb % (QK_K / 32);
    const uint8_t *     qs  = blk->qs + 8 * ib8;
    const int           qh  = blk->qh[ib8];
    const float         d   = (float) blk->d * (1 + 2 * ((blk->scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
        const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
        const int      signs = blk->signs[4 * ib8 + il];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float g1a = (float) ((grid1 >> (16 * j + 0)) & 0xff);
            const float g1b = (float) ((grid1 >> (16 * j + 8)) & 0xff);
            const float g2a = (float) ((grid2 >> (16 * j + 0)) & 0xff);
            const float g2b = (float) ((grid2 >> (16 * j + 8)) & 0xff);
            const int   s   = 2 * j;
            a[4 * il + j]     = E::make(d * ((signs & (1 << (s + 0))) ? -g1a : g1a),
                                        d * ((signs & (1 << (s + 1))) ? -g1b : g1b));
            a[4 * il + j + 2] = E::make(d * ((signs & (1 << (s + 4))) ? -g2a : g2a),
                                        d * ((signs & (1 << (s + 5))) ? -g2b : g2b));
        }
    }
}

// values per stored block, so a row of K values is K/qk blocks
template <typename block_q_t> struct fg_block_traits;
template <> struct fg_block_traits<block_iq4_nl> { static constexpr int qk = QK4_NL; };
template <> struct fg_block_traits<block_iq3_s>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq3_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq4_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xxs> { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_xs>  { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq2_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_s>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_iq1_m>   { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q8_0>    { static constexpr int qk = QK8_0; };
template <> struct fg_block_traits<block_q4_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q5_K>    { static constexpr int qk = QK_K; };
template <> struct fg_block_traits<block_q6_K>    { static constexpr int qk = QK_K; };

// The A stages below are the dequantize_block_iq* kernels rewritten for one k step. There a
// work-item handled one quarter (il) of one 32-wide sub-block (ib); here one lane produces the
// whole step, so il becomes a loop and ib is kb inside the superblock. Each quarter yields 8
// consecutive values, i.e. 4 pairs at a[4*il], so nothing larger than 8 floats is ever live.
#define FG_SUPERBLOCK(T)                                                         \
    static_assert(QK_K == 256, "the " #T " A stage assumes 8 sub-blocks per superblock"); \
    const T * blk = xrow + kb / (QK_K / 32);                                     \
    const int ib  = kb % (QK_K / 32)

template <typename E>
static __dpct_inline__ void fg_pack_quarter(const float * __restrict__ t, typename E::pair * a, int il) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[4 * il + j] = E::make(t[2 * j], t[2 * j + 1]);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq4_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq4_xs);
    // low nibbles fill the first half of the step, high nibbles the second, so the two halves
    // land at a[0..7] and a[8..15] and no quarter loop is needed
    const float d = (float) blk->d *
        ((((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((blk->scales_h >> (2 * ib)) & 3) << 4)) - 32);
    const uint8_t * q4 = blk->qs + 16 * ib;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        a[j]     = E::make(d * kvalues_iq4nl[q4[2 * j] & 0xf], d * kvalues_iq4nl[q4[2 * j + 1] & 0xf]);
        a[8 + j] = E::make(d * kvalues_iq4nl[q4[2 * j] >> 4],  d * kvalues_iq4nl[q4[2 * j + 1] >> 4]);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq3_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq3_xxs);
    const uint8_t *  q3    = blk->qs + 8 * ib;
    const uint16_t * gas   = (const uint16_t *) (blk->qs + QK_K / 4) + 2 * ib;
    const uint32_t   aux32 = gas[0] | (gas[1] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            t[j + 0] = d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            t[j + 4] = d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xxs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xxs);
    const uint16_t * q2    = blk->qs + 4 * ib;
    const uint8_t *  aux8  = (const uint8_t *) q2;
    const uint32_t   aux32 = q2[2] | (q2[3] << 16);
    const float      d     = (float) blk->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[il]);
        const uint8_t   signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_xs * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_xs);
    const uint16_t * q2 = blk->qs + 4 * ib;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid  = (const uint8_t *) (iq2xs_grid + (q2[il] & 511));
        const float     d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t   signs = ksigns_iq2xs[q2[il] >> 9];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq2_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq2_s);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint8_t * grid =
            (const uint8_t *) (iq2s_grid + (blk->qs[4 * ib + il] | ((blk->qh[ib] << (8 - 2 * il)) & 0x300)));
        const float   d     = (float) blk->d * (0.5f + ((blk->scales[ib] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        const uint8_t signs = blk->qs[QK_K / 8 + 4 * ib + il];
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_s * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_s);
    const float delta = blk->qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d     = (float) blk->d * (2 * ((blk->qh[ib] >> 12) & 7) + 1);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] | (((blk->qh[ib] >> (3 * il)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_iq1_m * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_iq1_m);
    const uint16_t * sc = (const uint16_t *) blk->scales;
    iq1m_scale_t     scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const int   ib16  = 2 * ib + il / 2;
        const float d     = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);
        const float delta = blk->qh[2 * ib + il / 2] & (0x08 << (4 * (il % 2))) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
        uint32_t       grid32[2];
        const int8_t * q = (const int8_t *) grid32;
        grid32[0] = iq1s_grid_gpu[blk->qs[4 * ib + il] |
                                  (((blk->qh[2 * ib + il / 2] >> (4 * (il % 2))) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        float t[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            t[j] = d * (q[j] + delta);
        }
        fg_pack_quarter<E>(t, a, il);
    }
}


// q8_0 and the k-quants. Each decode takes the fields of one 32-value sub-block, so the canonical
// layout and the reorder (SoA) layout of reorder_qw() share it and differ only in where the fields
// live, as in dequantize.hpp.
template <typename E>
static __dpct_inline__ void fg_decode_q8_0(const int8_t * __restrict__ qs, const float d, typename E::pair * a) {
#pragma unroll
    for (int j = 0; j < QK8_0 / 2; ++j) {
        a[j] = E::make(d * qs[2 * j], d * qs[2 * j + 1]);
    }
}

// same unpack as get_scale_min_k4() in dequantize.hpp
static __dpct_inline__ void fg_scale_min_k4(const int j, const uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// q4_K sub-block ib (0..7) uses scale/min pair ib and the low (even ib) or high (odd ib) nibbles of
// qs[32 * (ib / 2) ...], as in dequantize_row_q4_K
template <typename E>
static __dpct_inline__ void fg_decode_q4_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                                           const sycl::half2 dm, const int ib, typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    const float     d     = (float) dm[0] * sc;
    const float     m     = (float) dm[1] * mb;
    const uint8_t * q     = qs + 32 * (ib / 2);
    const int       shift = 4 * (ib % 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        a[j] = E::make(d * ((q[2 * j] >> shift) & 0xF) - m, d * ((q[2 * j + 1] >> shift) & 0xF) - m);
    }
}

// q5_K: q4_K plus one high bit per value, bit ib of qh[l]
template <typename E>
static __dpct_inline__ void fg_decode_q5_K(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ qh,
                                           const uint8_t * __restrict__ scales, const sycl::half2 dm, const int ib,
                                           typename E::pair * a) {
    uint8_t sc, mb;
    fg_scale_min_k4(ib, scales, sc, mb);
    const float     d     = (float) dm[0] * sc;
    const float     m     = (float) dm[1] * mb;
    const uint8_t * q     = qs + 32 * (ib / 2);
    const int       shift = 4 * (ib % 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int l = 2 * j;
        a[j] = E::make(d * (((q[l] >> shift) & 0xF) | (((qh[l] >> ib) & 1) << 4)) - m,
                       d * (((q[l + 1] >> shift) & 0xF) | (((qh[l + 1] >> ib) & 1) << 4)) - m);
    }
}

// q6_K: sub-block ib is quarter r = ib % 4 of half h = ib / 4. The half selects ql + 64h, qh + 32h
// and scales + 8h; the quarter selects ql + 32(r & 1), the ql nibble r / 2, the qh bit pair r and
// scales + 2r, as in dequantize_row_q6_K. The scale changes at value 16 of the sub-block.
template <typename E>
static __dpct_inline__ void fg_decode_q6_K(const uint8_t * __restrict__ ql, const uint8_t * __restrict__ qh,
                                           const int8_t * __restrict__ scales, const float d, const int ib,
                                           typename E::pair * a) {
    const int       h  = ib / 4;
    const int       r  = ib % 4;
    const uint8_t * q  = ql + 64 * h + 32 * (r & 1);
    const uint8_t * hb = qh + 32 * h;
    const int8_t *  sc = scales + 8 * h + 2 * r;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int   l  = 2 * j;
        const float dl = d * sc[j / 8];
        const int   q0 = (((q[l] >> (4 * (r / 2))) & 0xF) | (((hb[l] >> (2 * r)) & 3) << 4)) - 32;
        const int   q1 = (((q[l + 1] >> (4 * (r / 2))) & 0xF) | (((hb[l + 1] >> (2 * r)) & 3) << 4)) - 32;
        a[j] = E::make(dl * q0, dl * q1);
    }
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q8_0 * __restrict__ xrow, const int kb, typename E::pair * a) {
    fg_decode_q8_0<E>(xrow[kb].qs, (float) xrow[kb].d, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q4_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q4_K);
    fg_decode_q4_K<E>(blk->qs, blk->scales, blk->dm, ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q5_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q5_K);
    fg_decode_q5_K<E>(blk->qs, blk->qh, blk->scales, blk->dm, ib, a);
}

template <typename E>
static __dpct_inline__ void fg_stage_a(const block_q6_K * __restrict__ xrow, const int kb, typename E::pair * a) {
    FG_SUPERBLOCK(block_q6_K);
    fg_decode_q6_K<E>(blk->ql, blk->qh, blk->scales, (float) blk->d, ib, a);
}

// Reorder (SoA) layout: each block field is one stream over the nblocks of the matrix (of the expert
// slice for MUL_MAT_ID), in the order reorder_qw() writes them. Block ib holds k step kb.
template <typename block_q_t> struct fg_soa {
    static constexpr bool supported = false;
};

template <> struct fg_soa<block_q8_0> {
    static constexpr bool supported = true;
    // [qs][d]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int,
                                      typename E::pair * a) {
        const float d = (float) ((const sycl::half *) (x + nblocks * QK8_0))[ib];
        fg_decode_q8_0<E>((const int8_t *) x + ib * QK8_0, d, a);
    }
};

template <> struct fg_soa<block_q4_K> {
    static constexpr bool supported = true;
    // [qs][scales][dm]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t *   scales = x + nblocks * (QK_K / 2);
        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q4_K<E>(x + ib * (QK_K / 2), scales + ib * K_SCALE_SIZE, dm, kb % (QK_K / 32), a);
    }
};

template <> struct fg_soa<block_q5_K> {
    static constexpr bool supported = true;
    // [qs][qh][scales][dm]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t *   qh     = x + nblocks * (QK_K / 2);
        const uint8_t *   scales = qh + nblocks * (QK_K / 8);
        const sycl::half2 dm     = ((const sycl::half2 *) (scales + nblocks * K_SCALE_SIZE))[ib];
        fg_decode_q5_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 8), scales + ib * K_SCALE_SIZE, dm,
                          kb % (QK_K / 32), a);
    }
};

template <> struct fg_soa<block_q6_K> {
    static constexpr bool supported = true;
    // [ql][qh][scales][d]
    template <typename E>
    static __dpct_inline__ void stage(const uint8_t * x, const size_t nblocks, const size_t ib, const int kb,
                                      typename E::pair * a) {
        const uint8_t * qh     = x + nblocks * (QK_K / 2);
        const uint8_t * scales = qh + nblocks * (QK_K / 4);
        const float     d      = (float) ((const sycl::half *) (scales + nblocks * (QK_K / 16)))[ib];
        fg_decode_q6_K<E>(x + ib * (QK_K / 2), qh + ib * (QK_K / 4), (const int8_t *) scales + ib * (QK_K / 16), d,
                          kb % (QK_K / 32), a);
    }
};


// one SG_ROWS x BN output tile: B columns [b0, b0 + BN) of packed_b go to dst columns [n0, n1),
// n1 - n0 <= BN
template <typename S, typename block_q_t, bool reordered>
static void fused_dequant_gemm_tile(
    const block_q_t * __restrict__ x,
    const typename S::tsb * __restrict__ packed_b,
    float * __restrict__ dst,
    const int M, const int Npad, const int K, const int ldd,
    const int b0, const int n0, const int n1,
    sycl::local_accessor<typename S::tsa, 1> tile_a,
    sycl::local_accessor<float, 1> tile_c,
    const sycl::nd_item<2> & item) {
    if constexpr (fg_built<S::SG>()) {
        using EA = typename S::EA;
        using TA = typename S::ta;
        using TB = typename S::tb;
        const auto sg     = item.get_sub_group();
        const int  sg_id  = sg.get_group_id()[0];
        const int  lane   = sg.get_local_id()[0];
        const int  m0     = item.get_group(1) * S::SG_ROWS;
        const int  nstep  = K / FG_BK;
        const int  a_base = sg_id * S::SG_ROWS * FG_BK;
        const int  c_base = sg_id * S::SG_ROWS * S::BN;

        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, S::TM, S::TN> acc[S::MT][S::NT];
#pragma unroll
        for (int mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_fill(sg, acc[mt][nt], 0.0f);
            }
        }

        // lane decodes rows lane, lane + SG, ... of the sub-group's SG_ROWS
        constexpr int         KPB     = fg_block_traits<block_q_t>::qk / FG_BK; // k steps per block
        const size_t          bpr     = K / fg_block_traits<block_q_t>::qk;
        const size_t          nblocks = (size_t) M * bpr;
        size_t                row_blk[S::RPL];
        bool                  row_ok[S::RPL];
        typename EA::pair *   a[S::RPL];
#pragma unroll
        for (int r = 0; r < S::RPL; ++r) {
            const int row = m0 + r * S::SG + lane;
            row_ok[r]  = row < M;
            row_blk[r] = (size_t) (row_ok[r] ? row : 0) * bpr;
            a[r]       = (typename EA::pair *) &tile_a[a_base + (r * S::SG + lane) * FG_BK];
        }

        const auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                    sycl::access::decorated::no>(packed_b);
        const int b_stride = Npad * S::VNNI;

        const int kb_begin = (sg_id * nstep) / FG_KSPLIT;
        const int kb_end   = ((sg_id + 1) * nstep) / FG_KSPLIT;
        for (int kb = kb_begin; kb < kb_end; ++kb) {
#pragma unroll
            for (int r = 0; r < S::RPL; ++r) {
                if (row_ok[r]) {
                    if constexpr (reordered) {
                        fg_soa<block_q_t>::template stage<EA>((const uint8_t *) x, nblocks, row_blk[r] + kb / KPB, kb,
                                                              a[r]);
                    } else {
                        fg_stage_a<EA>(x + row_blk[r], kb, a[r]);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < FG_BK / 2; ++j) {
                        a[r][j] = EA::make(0.0f, 0.0f);
                    }
                }
            }
            sycl::group_barrier(sg);

#pragma unroll
            for (int kt = 0; kt < FG_BK / S::TK; ++kt) {
                const int kq0 = (kb * FG_BK + kt * S::TK) / S::VNNI;
                mx::joint_matrix<sycl::sub_group, TB, mx::use::b, S::TK, S::TN, S::b_layout> sub_b[S::NT];
#pragma unroll
                for (int nt = 0; nt < S::NT; ++nt) {
                    mx::joint_matrix_load(sg, sub_b[nt], b_ptr + (size_t) kq0 * b_stride + (b0 + nt * S::TN) * S::VNNI, b_stride);
                }
#pragma unroll
                for (int mt = 0; mt < S::MT; ++mt) {
                    mx::joint_matrix<sycl::sub_group, TA, mx::use::a, S::TM, S::TK, mx::layout::row_major> sub_a;
                    mx::joint_matrix_load(sg, sub_a,
                        tile_a.template get_multi_ptr<sycl::access::decorated::no>() + a_base + (mt * S::TM) * FG_BK + kt * S::TK,
                        FG_BK);
#pragma unroll
                    for (int nt = 0; nt < S::NT; ++nt) {
                        mx::joint_matrix_mad(sg, acc[mt][nt], sub_a, sub_b[nt], acc[mt][nt]);
                    }
                }
            }
            // the next step overwrites tile_a
            sycl::group_barrier(sg);
        }

#pragma unroll
        for (int mt = 0; mt < S::MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < S::NT; ++nt) {
                mx::joint_matrix_store(sg, acc[mt][nt],
                    tile_c.template get_multi_ptr<sycl::access::decorated::no>() + c_base + (mt * S::TM) * S::BN + nt * S::TN,
                    S::BN, mx::layout::row_major);
            }
        }
        sycl::group_barrier(item.get_group());

        // sum the K splits; consecutive lanes write consecutive rows of one dst column
        for (int idx = item.get_local_linear_id(); idx < S::SG_ROWS * S::BN; idx += S::WG_SIZE) {
            const int r = idx % S::SG_ROWS;
            const int c = idx / S::SG_ROWS;
            const int m = m0 + r;
            const int n = n0 + c;
            if (m < M && n < n1) {
                float sum = 0.0f;
#pragma unroll
                for (int s = 0; s < FG_KSPLIT; ++s) {
                    sum += tile_c[s * S::SG_ROWS * S::BN + r * S::BN + c];
                }
                dst[(size_t) n * ldd + m] = sum;
            }
        }
    }
}

template <typename S, typename block_q_t>
static void fused_dequant_gemm_launch(const void * src0, const typename S::tsb * packed, float * dst, const int M,
                                      const int N, const int Npad, const int K, const int ldd,
                                      const int64_t groups_n, const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups_n, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int n0 = item.get_group(0) * S::BN;
                fused_dequant_gemm_tile<S, block_q_t, false>((const block_q_t *) src0, packed, dst, M, Npad, K, ldd,
                                                      n0, n0, N, tile_a, tile_c, item);
            });
    });
}

// grouped: work-group (t, mt) is tile t of the schedule; its B columns sit at t * BN
template <typename S, typename block_q_t, bool reordered>
static void grouped_dequant_gemm_launch(const char * src0_dd, const size_t expert_stride,
                                        const ggml_sycl_gg_tile * tiles_ptr, const typename S::tsb * packed, float * dst,
                                        const int M, const int Npad, const int K, const int64_t n_tiles,
                                        const int64_t groups_m, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<typename S::tsa, 1> tile_a(FG_KSPLIT * S::SG_ROWS * FG_BK, cgh);
        sycl::local_accessor<float, 1>          tile_c(FG_KSPLIT * S::SG_ROWS * S::BN, cgh);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(n_tiles, groups_m * S::WG_SIZE), sycl::range<2>(1, S::WG_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(S::SG)]] {
                const int               t    = item.get_group(0);
                const ggml_sycl_gg_tile tile = tiles_ptr[t];
                const block_q_t *       x    = (const block_q_t *) (src0_dd + (size_t) tile.expert * expert_stride);
                fused_dequant_gemm_tile<S, block_q_t, reordered>(x, packed, dst, M, Npad, K, M, t * S::BN, tile.n0,
                                                                 tile.n1, tile_a, tile_c, item);
            });
    });
}

// src1 f32 rows -> packed [K/V][n_tiles*BN][V], tile t holds its rows [n0, n1) at columns t*BN..,
// zero past n1. The column runs fastest so a sub-group writes one contiguous run.
template <typename S>
static void grouped_gemm_pack_b(const float * y, typename S::tsb * packed, const ggml_sycl_gg_tile * tiles, int Npad,
                                int K, dpct::queue_ptr stream) {
    using E        = typename S::EB;
    constexpr int V = S::VNNI;
    const int kqs   = K / V;
    stream->parallel_for(sycl::range<1>((size_t) Npad * kqs), [=](sycl::id<1> id) {
        const size_t idx = id[0];
        const int    kq  = idx / Npad;
        const int    n   = idx - (size_t) kq * Npad;
        const ggml_sycl_gg_tile tile = tiles[n / S::BN];
        const int    row = tile.n0 + n % S::BN;
        // one guarded load run per work-item, as a per-element select costs ~1.5% prefill
        typename S::tsb vals[V] = {};
        if (row < tile.n1) {
            const float * src = y + (size_t) row * K + V * kq;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                vals[v] = E::cvt(src[v]);
            }
        }
        typename S::tsb * out = packed + ((size_t) kq * Npad + n) * V;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            out[v] = vals[v];
        }
    });
}

// q8_0 and the k-quants take only the grouped path. The plain kernel decodes A again for every BN columns
// of a dense batch, and for these formats that costs more than the one dequantization of the library GEMM.
template <typename T> static constexpr bool fg_plain_ok() {
    return !std::is_same_v<T, block_q8_0> && !std::is_same_v<T, block_q4_K> && !std::is_same_v<T, block_q5_K> &&
           !std::is_same_v<T, block_q6_K>;
}

template <typename T, bool R> struct fg_tag {
    using type                     = T;
    static constexpr bool reordered = R;
};

// calls f(fg_tag<block_q_t, reordered>{}) for the weight format and layout; false if it has no A stage
template <typename T, typename F> static bool fg_visit_layout(bool reordered, F && f) {
    if (!reordered) {
        f(fg_tag<T, false>{});
        return true;
    }
    if constexpr (fg_soa<T>::supported) {
        f(fg_tag<T, true>{});
        return true;
    }
    return false;
}

template <typename F> static bool fg_visit_type(ggml_type type, bool reordered, F && f) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:  return fg_visit_layout<block_iq4_nl>(reordered, f);
        case GGML_TYPE_IQ3_S:   return fg_visit_layout<block_iq3_s>(reordered, f);
        case GGML_TYPE_IQ4_XS:  return fg_visit_layout<block_iq4_xs>(reordered, f);
        case GGML_TYPE_IQ3_XXS: return fg_visit_layout<block_iq3_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XXS: return fg_visit_layout<block_iq2_xxs>(reordered, f);
        case GGML_TYPE_IQ2_XS:  return fg_visit_layout<block_iq2_xs>(reordered, f);
        case GGML_TYPE_IQ2_S:   return fg_visit_layout<block_iq2_s>(reordered, f);
        case GGML_TYPE_IQ1_S:   return fg_visit_layout<block_iq1_s>(reordered, f);
        case GGML_TYPE_IQ1_M:   return fg_visit_layout<block_iq1_m>(reordered, f);
        case GGML_TYPE_Q8_0:    return fg_visit_layout<block_q8_0>(reordered, f);
        case GGML_TYPE_Q4_K:    return fg_visit_layout<block_q4_K>(reordered, f);
        case GGML_TYPE_Q5_K:    return fg_visit_layout<block_q5_K>(reordered, f);
        case GGML_TYPE_Q6_K:    return fg_visit_layout<block_q6_K>(reordered, f);
        default:                return false;
    }
}

template <typename S>
static bool fg_fused_run(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
                         float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                         dpct::queue_ptr stream) {
    const int64_t groups_n = (N + S::BN - 1) / S::BN;
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (groups_n * S::BN);

    // src1 is read in its own type: one pass, converted in registers only if B differs
    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * Npad);
    if (src1_type == GGML_TYPE_F16) {
        fused_gemm_pack_b<typename S::EB>((const sycl::half *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else if (src1_type == GGML_TYPE_BF16) {
        fused_gemm_pack_b<typename S::EB>((const fg_bf16 *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    } else {
        fused_gemm_pack_b<typename S::EB>((const float *) src1, packed_b.get(), (int) N, Npad, (int) K, stream);
    }

    const typename S::tsb * packed = packed_b.get();
    return fg_visit_type(src0_type, false, [&](auto tag) {
        using block_q_t = typename decltype(tag)::type;
        if constexpr (fg_plain_ok<block_q_t>()) {
            fused_dequant_gemm_launch<S, block_q_t>(src0, packed, dst, (int) M, (int) N, Npad, (int) K, (int) ldd,
                                                    groups_n, groups_m, stream);
        }
    });
}

bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
                                  int32_t src1_prec, float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    // every BN columns dequantize A again, so wide N is left to the library GEMM
    bool plain_ok = false;
    fg_visit_type(src0_type, false, [&](auto tag) { plain_ok = fg_plain_ok<typename decltype(tag)::type>(); });
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !plain_ok) {
        return false;
    }
    if (src1_type != GGML_TYPE_F32 && src1_type != GGML_TYPE_F16 && src1_type != GGML_TYPE_BF16) {
        return false;
    }
    if (!ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, N, K, ldd)) {
        return false;
    }
    const int combo = fg_pick_combo(stream, src1_type, src1_prec);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_fused_run<decltype(s)>(src0_type, src0, src1, src1_type, dst, M, N, K, ldd, pool, stream);
    });
    return launched;
}

template <typename S>
static bool fg_grouped_run(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                           const float * src1, float * dst, const int64_t * expert_row_offsets, int64_t n_as, int64_t M,
                           int64_t K, std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                           dpct::queue_ptr stream) {
    // the host knows every slice, so it lays out the work-groups: no search on the device
    tiles.clear();
    for (int64_t e = 0; e < n_as; ++e) {
        const int64_t end = expert_row_offsets[e + 1];
        for (int64_t n0 = expert_row_offsets[e]; n0 < end; n0 += S::BN) {
            tiles.push_back({ (int32_t) e, (int32_t) n0, (int32_t) std::min<int64_t>(n0 + S::BN, end) });
        }
    }
    const int64_t n_tiles  = tiles.size();
    const int64_t groups_m = (M + S::SG_ROWS - 1) / S::SG_ROWS;
    const int     Npad     = (int) (n_tiles * S::BN);

    const int64_t max_tiles = grouped_gemm_max_tiles(expert_row_offsets[n_as], n_as, S::BN);
    GGML_ASSERT(n_tiles <= max_tiles);
    ggml_sycl_pool_alloc<ggml_sycl_gg_tile> tiles_dev(pool, max_tiles);
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(tiles_dev.get(), tiles.data(), n_tiles * sizeof(ggml_sycl_gg_tile))));

    ggml_sycl_pool_alloc<typename S::tsb> packed_b(pool, (size_t) K * max_tiles * S::BN);
    grouped_gemm_pack_b<S>(src1, packed_b.get(), tiles_dev.get(), Npad, (int) K, stream);

    const typename S::tsb *   packed    = packed_b.get();
    const ggml_sycl_gg_tile * tiles_ptr = tiles_dev.get();
    const char *              src0_dd   = (const char *) src0_base;
    return fg_visit_type(src0_type, reordered, [&](auto tag) {
        using T = decltype(tag);
        grouped_dequant_gemm_launch<S, typename T::type, T::reordered>(src0_dd, expert_stride, tiles_ptr, packed, dst,
                                                                       (int) M, Npad, (int) K, n_tiles, groups_m,
                                                                       stream);
    });
}

// Large-batch Q4_K/Q5_K MoE prefill path adapted from the engine's xmxg::iq_xmx_grouped
// implementation (engine/sycl/src/kernels/cuda/iq_kernels.dp.cpp).
// Unlike the small grouped path above,
// this uses 64/128-row x 128-output tiles and decodes the GGUF weights directly into SLM. It is
// intended for the large-row regime where the existing grouped path deliberately declines and
// the caller otherwise launches a separate library GEMM for each expert.
namespace fg_xmxg {
#if !defined(GGML_SYCL_XMX_AOT_SG) || GGML_SYCL_XMX_AOT_SG == 8
constexpr int SG = 8;
constexpr int MAX_EXPERTS = 16;
struct group {
    const uint8_t * w[MAX_EXPERTS];
    int32_t off[MAX_EXPERTS + 1];
    int32_t tstart[MAX_EXPERTS + 1];
    int32_t n;
};

// One batch of at most MAX_EXPERTS experts. The local-reduction experiment writes
// this batch's compact expert products to a reusable scratch buffer, then reduces
// the rows for each touched token without global floating-point atomics.
struct local_reduce_batch {
    group experts;
    int32_t row_begin;
    int32_t row_end;
    int32_t token_begin;
    int32_t token_count;
};
template <int SGM_, int SGN_, int MT_, int NT_> struct config {
    static constexpr int SGM = SGM_, SGN = SGN_, MT = MT_, NT = NT_;
    static constexpr int KS = 32, BM = SGM * MT * 8, BN = SGN * NT * SG, WG = SGM * SGN * SG;
};
using small_config = config<2, 8, 4, 2>; // 64 x 128, 16 subgroups of 8
using large_config = config<4, 4, 4, 4>; // 128 x 128, 16 subgroups of 8

static inline void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <typename T, bool REORDERED, bool Q5>
static inline void decode8(const uint8_t * w, int64_t blocks, int64_t ibs, int ib32, int sub, T * dst) {
    const uint8_t * qs;
    const uint8_t * qh = nullptr;
    const uint8_t * scales;
    sycl::half2 dm;
    if constexpr (Q5) {
        if constexpr (REORDERED) {
            const size_t ql_bytes = (size_t) blocks * (QK_K / 2);
            const size_t qh_bytes = (size_t) blocks * (QK_K / 8);
            qs = w + (size_t) ibs * (QK_K / 2);
            qh = w + ql_bytes + (size_t) ibs * (QK_K / 8);
            scales = w + ql_bytes + qh_bytes + (size_t) ibs * K_SCALE_SIZE;
            dm = ((const sycl::half2 *) (w + ql_bytes + qh_bytes + (size_t) blocks * K_SCALE_SIZE))[ibs];
        } else {
            const block_q5_K * b = (const block_q5_K *) w + ibs;
            qs = b->qs;
            qh = b->qh;
            scales = b->scales;
            dm = b->dm;
        }
    } else {
        if constexpr (REORDERED) {
            const size_t q_bytes = (size_t) blocks * (QK_K / 2);
            qs = w + (size_t) ibs * (QK_K / 2);
            scales = w + q_bytes + (size_t) ibs * K_SCALE_SIZE;
            dm = ((const sycl::half2 *) (w + q_bytes + (size_t) blocks * K_SCALE_SIZE))[ibs];
        } else {
            const block_q4_K * b = (const block_q4_K *) w + ibs;
            qs = b->qs;
            scales = b->scales;
            dm = b->dm;
        }
    }
    uint8_t sc, m;
    scale_min_k4(ib32, scales, sc, m);
    const float d = (float) dm[0] * sc;
    const float min = (float) dm[1] * m;
    const uint8_t * ql = qs + 32 * (ib32 >> 1) + 8 * sub;
    const uint8_t * qh8 = nullptr;
    if constexpr (Q5) qh8 = qh + 8 * sub;
    const uint8_t hm = (uint8_t) (1u << ib32);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int q = (ib32 & 1) ? (ql[j] >> 4) : (ql[j] & 0x0f);
        if constexpr (Q5) q |= (qh8[j] & hm) ? 16 : 0;
        dst[j] = (T) (d * q - min);
    }
}

template <class C>
static int fill_tiles(group & gp) {
    int t = 0;
    for (int i = 0; i < gp.n; ++i) {
        gp.tstart[i] = t;
        const int rows = gp.off[i + 1] - gp.off[i];
        t += (rows + C::BM - 1) / C::BM;
    }
    gp.tstart[gp.n] = t;
    return t;
}

template <typename T, bool REORDERED, bool Q5, bool SUMMED, bool LOCAL_REDUCE> class small_kernel;
template <typename T, bool REORDERED, bool Q5, bool SUMMED, bool LOCAL_REDUCE> class large_kernel;
template <class C, class Name, typename T, bool REORDERED, bool Q5, bool SUMMED, bool LOCAL_REDUCE>
static void launch(sycl::queue & q, const group & gp, const T * X, float * Y, int blocks,
                   int ntiles, int N, int K, const mmid_row_mapping * row_mapping,
                   const float * route_weights, int n_ids, float * sum_dst, int group_row_base) {
    constexpr int BM = C::BM, BN = C::BN, KS = C::KS, WG = C::WG, TN = SG;
    constexpr int LDA = KS + 8, LDB = KS + 8;
    const int nb = N / BN;
    auto props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler & h) {
        sycl::local_accessor<T, 1> sa(BM * LDA, h), sb(BN * LDB, h);
        h.parallel_for<Name>(sycl::nd_range<2>({(size_t) ntiles, (size_t) nb * WG}, {1, (size_t) WG}), props,
                             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int t = (int) it.get_group(0);
            int e = 0;
            while (e + 1 < gp.n && t >= gp.tstart[e + 1]) ++e;
            const int m0 = (t - gp.tstart[e]) * BM;
            const int rows = sycl::min(BM, gp.off[e + 1] - gp.off[e] - m0);
            const T * Xe = X + ((size_t) gp.off[e] + m0) * K;
            float * Ye = nullptr;
            if constexpr (!SUMMED || LOCAL_REDUCE) {
                const size_t row = (size_t) gp.off[e] + (size_t) m0;
                const size_t row_offset = LOCAL_REDUCE ? row - (size_t) group_row_base : row;
                Ye = Y + row_offset * (size_t) N;
            }
            const uint8_t * w = gp.w[e];
            const int n0 = (int) it.get_group(1) * BN;
            const int lid = (int) it.get_local_id(1), sgid = lid / SG;
            const int sgm = sgid / C::SGN, sgn = sgid % C::SGN;
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, TN> acc[C::MT][C::NT];
            for (int i = 0; i < C::MT; ++i)
                for (int j = 0; j < C::NT; ++j) mx::joint_matrix_fill(sg, acc[i][j], 0.0f);
            T * pa = sa.template get_multi_ptr<sycl::access::decorated::no>().get();
            T * pb = sb.template get_multi_ptr<sycl::access::decorated::no>().get();
            for (int k0 = 0; k0 < K; k0 += KS) {
                constexpr int VA = BM * KS / 8;
                for (int v = lid; v < VA; v += WG) {
                    const int r = v / (KS / 8), c = (v % (KS / 8)) * 8;
                    if (r < rows) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) pa[(size_t) r * LDA + c + j] = Xe[(size_t) r * K + k0 + c + j];
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) pa[(size_t) r * LDA + c + j] = (T) 0.0f;
                    }
                }
                // Decode each output row's next 32 quantized values directly into the B tile.
                for (int v = lid; v < BN * 4; v += WG) {
                    const int r = v >> 2, sub = v & 3, nr = n0 + r;
                    const int64_t flat = (int64_t) nr * K + k0;
                    const int64_t ibs = flat >> 8;
                    const int ib32 = (int) ((flat & 255) >> 5);
                    decode8<T, REORDERED, Q5>(w, blocks, ibs, ib32, sub, pb + (size_t) r * LDB + 8 * sub);
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int kk = 0; kk < KS; kk += 16) {
                    mx::joint_matrix<sycl::sub_group, T, mx::use::a, 8, 16, mx::layout::row_major> am[C::MT];
                    mx::joint_matrix<sycl::sub_group, T, mx::use::b, 16, TN, mx::layout::col_major> bm[C::NT];
                    for (int i = 0; i < C::MT; ++i)
                        mx::joint_matrix_load(sg, am[i], sa.template get_multi_ptr<sycl::access::decorated::no>() +
                                                   (sgm * C::MT * 8 + i * 8) * LDA + kk, LDA);
                    for (int j = 0; j < C::NT; ++j)
                        mx::joint_matrix_load(sg, bm[j], sb.template get_multi_ptr<sycl::access::decorated::no>() +
                                                   (sgn * C::NT * TN + j * TN) * LDB + kk, LDB);
                    for (int i = 0; i < C::MT; ++i)
                        for (int j = 0; j < C::NT; ++j) mx::joint_matrix_mad(sg, acc[i][j], am[i], bm[j], acc[i][j]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (int i = 0; i < C::MT; ++i) {
                const int r = sgm * C::MT * 8 + i * 8;
                if (r >= rows) continue;
                for (int j = 0; j < C::NT; ++j) {
                    const int c = n0 + sgn * C::NT * TN + j * TN;
                    if constexpr (SUMMED && !LOCAL_REDUCE) {
                        // Atomic baseline: different expert workgroups may contribute to the
                        // same token/output element, so the final accumulation is global.
                        float * st = reinterpret_cast<float *>(pa) + sgid * 8 * TN;
                        mx::joint_matrix_store(sg, acc[i][j],
                            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(st),
                            TN, mx::layout::row_major);
                        sycl::group_barrier(sg);
                        const int lim = sycl::min(8, rows - r);
                        const int ln = (int) sg.get_local_linear_id();
                        for (int rr = 0; rr < lim; ++rr) {
                            const int compact_row = gp.off[e] + m0 + r + rr;
                            const mmid_row_mapping mapping = row_mapping[compact_row];
                            const float weight = route_weights[(int64_t) mapping.i2 * n_ids + mapping.i1];
                            const float value = st[rr * TN + ln] * weight;
                            float * out = sum_dst + (int64_t) mapping.i2 * N + c + ln;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space> atom(*out);
                            atom.fetch_add(value);
                        }
                        sycl::group_barrier(sg);
                    } else if (r + 8 <= rows) {
                        mx::joint_matrix_store(sg, acc[i][j],
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                Ye + (size_t) r * N + c), N, mx::layout::row_major);
                    } else {
                        float * st = reinterpret_cast<float *>(pa) + sgid * 8 * TN;
                        mx::joint_matrix_store(sg, acc[i][j],
                            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(st),
                            TN, mx::layout::row_major);
                        sycl::group_barrier(sg);
                        const int lim = rows - r, ln = (int) sg.get_local_linear_id();
                        for (int rr = 0; rr < lim; ++rr) Ye[(size_t) (r + rr) * N + c + ln] = st[rr * TN + ln];
                        sycl::group_barrier(sg);
                    }
                }
            }
        });
    });
}

template <class C, typename T, bool Q5, bool SUMMED, bool LOCAL_REDUCE = false>
static void launch_layout(sycl::queue & q, const group & gp, const T * X, float * Y, int blocks,
                          int ntiles, int N, int K, bool reordered,
                          const mmid_row_mapping * row_mapping, const float * route_weights,
                          int n_ids, float * sum_dst, int group_row_base = 0) {
    if (reordered) {
        if constexpr (std::is_same_v<C, small_config>) {
            launch<C, small_kernel<T, true, Q5, SUMMED, LOCAL_REDUCE>, T, true, Q5, SUMMED, LOCAL_REDUCE>(q, gp, X, Y, blocks, ntiles, N, K,
                row_mapping, route_weights, n_ids, sum_dst, group_row_base);
        } else {
            launch<C, large_kernel<T, true, Q5, SUMMED, LOCAL_REDUCE>, T, true, Q5, SUMMED, LOCAL_REDUCE>(q, gp, X, Y, blocks, ntiles, N, K,
                row_mapping, route_weights, n_ids, sum_dst, group_row_base);
        }
    } else {
        if constexpr (std::is_same_v<C, small_config>) {
            launch<C, small_kernel<T, false, Q5, SUMMED, LOCAL_REDUCE>, T, false, Q5, SUMMED, LOCAL_REDUCE>(q, gp, X, Y, blocks, ntiles, N, K,
                row_mapping, route_weights, n_ids, sum_dst, group_row_base);
        } else {
            launch<C, large_kernel<T, false, Q5, SUMMED, LOCAL_REDUCE>, T, false, Q5, SUMMED, LOCAL_REDUCE>(q, gp, X, Y, blocks, ntiles, N, K,
                row_mapping, route_weights, n_ids, sum_dst, group_row_base);
        }
    }
}
// Reduction kernel for one expert batch. Each work-item owns one final output element,
// scans only that token's top-k slots, and accumulates the partial rows belonging to this
// expert batch in registers. The queue is in-order, so batches update sum_dst serially.
static void reduce_group_partials(sycl::queue & q, const float * products, const int32_t * route_to_compact,
                                  const int32_t * group_tokens, int32_t group_token_count,
                                  int32_t group_row_begin, int32_t group_row_end,
                                  const float * route_weights, int32_t n_ids, int32_t M,
                                  float * sum_dst) {
    constexpr size_t WG = 128;
    const size_t items = (size_t) group_token_count * (size_t) M;
    const size_t global = ((items + WG - 1) / WG) * WG;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t flat = (size_t) it.get_global_linear_id();
        if (flat >= items) return;
        const int32_t token = group_tokens[flat / (size_t) M];
        const int32_t col = (int32_t) (flat % (size_t) M);
        float partial = 0.0f;
        for (int32_t slot = 0; slot < n_ids; ++slot) {
            const int32_t compact = route_to_compact[(size_t) token * (size_t) n_ids + (size_t) slot];
            if (compact >= group_row_begin && compact < group_row_end) {
                const float value = products[(size_t) (compact - group_row_begin) * (size_t) M + (size_t) col];
                const float weight = route_weights[(size_t) token * (size_t) n_ids + (size_t) slot];
                partial += value * weight;
            }
        }
        // Exactly one work-item owns each (token, output-column) for this batch.
        // Different expert batches are submitted sequentially on the in-order SYCL queue.
        sum_dst[(size_t) token * (size_t) M + (size_t) col] += partial;
    });
}

template <typename T, bool SUMMED, bool LOCAL_REDUCE = false>
static bool run_grouped(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                        const float * src1, float * dst, const int64_t * expert_row_offsets,
                        int64_t n_as, int64_t M, int64_t K, int64_t valid_rows,
                        int blocks, ggml_sycl_pool & pool, dpct::queue_ptr stream,
                        const mmid_row_mapping * row_mapping, const float * route_weights,
                        int n_ids, int64_t n_tokens, float * sum_dst,
                        const mmid_row_mapping * row_mapping_host = nullptr) {
    const uint64_t elements64 = (uint64_t) valid_rows * (uint64_t) K;
    if (elements64 > SIZE_MAX / sizeof(T)) return false;
    const size_t elements = (size_t) elements64;
    // Keep pool allocation handles in allocation order. The SYCL VMM pool is LIFO:
    // destruction must release group_tokens, group_products, route_map, then packed_x.
    ggml_sycl_pool_alloc<T> packed_x(pool, elements);
    ggml_sycl_pool_alloc<int32_t> route_to_compact_dev(pool);
    ggml_sycl_pool_alloc<float> group_products(pool);
    ggml_sycl_pool_alloc<int32_t> group_tokens_dev(pool);
    if (packed_x.get() == nullptr) {
        GGML_LOG_WARN("%s: unable to allocate %zu bytes for packed MoE activations; using fallback\n",
                      __func__, elements * sizeof(T));
        return false;
    }

    sycl::queue & q = *stream;
    std::vector<local_reduce_batch> local_batches;
    std::vector<int32_t> local_group_tokens;
    size_t max_group_rows = 0;
    if constexpr (LOCAL_REDUCE) {
        if constexpr (!SUMMED) {
            return false;
        } else {
            if (row_mapping_host == nullptr || row_mapping == nullptr || route_weights == nullptr ||
                sum_dst == nullptr || n_ids <= 0 || n_tokens <= 0 || n_tokens > INT32_MAX) {
                return false;
            }
            const uint64_t route_entries = (uint64_t) n_ids * (uint64_t) n_tokens;
            // The inverse route map is an additional host/device allocation. Keep the
            // experimental path bounded; large contexts fall back to the atomic kernel.
            if (route_entries > SIZE_MAX / sizeof(int32_t) ||
                route_entries * sizeof(int32_t) > 256u * 1024u * 1024u) {
                GGML_LOG_WARN("%s: local MoE route map exceeds 256 MiB; using atomic path\n", __func__);
                return false;
            }
            std::vector<int32_t> route_to_compact_host((size_t) route_entries, -1);
            for (int64_t r = 0; r < valid_rows; ++r) {
                const mmid_row_mapping mapping = row_mapping_host[r];
                if (mapping.i1 < 0 || mapping.i1 >= n_ids || mapping.i2 < 0 || mapping.i2 >= n_tokens) {
                    return false;
                }
                const size_t index = (size_t) mapping.i2 * (size_t) n_ids + (size_t) mapping.i1;
                if (route_to_compact_host[index] != -1) {
                    return false; // duplicate slot/token mapping: use the atomic path
                }
                route_to_compact_host[index] = (int32_t) r;
            }

            std::vector<uint8_t> token_seen((size_t) n_tokens, 0);
            int64_t e = 0;
            while (e < n_as) {
                group gp{};
                while (e < n_as && gp.n < MAX_EXPERTS) {
                    const int64_t first = expert_row_offsets[e];
                    const int64_t last  = expert_row_offsets[e + 1];
                    if (last > first) {
                        gp.w[gp.n] = (const uint8_t *) src0_base + (size_t) e * expert_stride;
                        gp.off[gp.n] = (int32_t) first;
                        gp.off[gp.n + 1] = (int32_t) last;
                        ++gp.n;
                    }
                    ++e;
                }
                if (gp.n == 0) continue;
                const int32_t row_begin = gp.off[0];
                const int32_t row_end = gp.off[gp.n];
                std::vector<int32_t> batch_tokens;
                for (int32_t r = row_begin; r < row_end; ++r) {
                    const int32_t token = row_mapping_host[r].i2;
                    if (token < 0 || token >= n_tokens) return false;
                    if (!token_seen[(size_t) token]) {
                        token_seen[(size_t) token] = 1;
                        batch_tokens.push_back(token);
                    }
                }
                for (int32_t token : batch_tokens) token_seen[(size_t) token] = 0;
                std::sort(batch_tokens.begin(), batch_tokens.end());
                if (local_group_tokens.size() > (size_t) INT32_MAX - batch_tokens.size()) return false;
                local_reduce_batch batch{};
                batch.experts = gp;
                batch.row_begin = row_begin;
                batch.row_end = row_end;
                batch.token_begin = (int32_t) local_group_tokens.size();
                batch.token_count = (int32_t) batch_tokens.size();
                local_group_tokens.insert(local_group_tokens.end(), batch_tokens.begin(), batch_tokens.end());
                local_batches.push_back(batch);
                max_group_rows = std::max(max_group_rows, (size_t) (row_end - row_begin));
            }
            if (local_batches.empty() || max_group_rows == 0 ||
                max_group_rows > SIZE_MAX / (size_t) M / sizeof(float) ||
                max_group_rows * (size_t) M * sizeof(float) > 256u * 1024u * 1024u ||
                local_group_tokens.empty() ||
                local_group_tokens.size() > (256u * 1024u * 1024u) / sizeof(int32_t)) {
                GGML_LOG_WARN("%s: local MoE reduction scratch exceeds limit or has invalid routing; using atomic path\n", __func__);
                return false;
            }
            route_to_compact_dev.alloc((size_t) n_ids * (size_t) n_tokens);
            group_products.alloc(max_group_rows * (size_t) M);
            group_tokens_dev.alloc(local_group_tokens.size());
            if (route_to_compact_dev.get() == nullptr || group_products.get() == nullptr || group_tokens_dev.get() == nullptr) {
                GGML_LOG_WARN("%s: unable to allocate local MoE reduction scratch; using atomic path\n", __func__);
                return false;
            }
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(route_to_compact_dev.get(), route_to_compact_host.data(),
                                                       route_to_compact_host.size() * sizeof(int32_t))));
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(group_tokens_dev.get(), local_group_tokens.data(),
                                                       local_group_tokens.size() * sizeof(int32_t))));
            // Both source vectors are temporary host storage. Complete these uploads before
            // they leave scope; the subsequent device kernels remain asynchronous/in-order.
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        }
    }
    if constexpr (SUMMED) {
        GGML_ASSERT(row_mapping != nullptr && route_weights != nullptr && sum_dst != nullptr && n_ids > 0 && n_tokens > 0);
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(sum_dst, 0, (size_t) M * (size_t) n_tokens * sizeof(float))));
    }
    T * packed = packed_x.get();
    // The engine's XMX kernel consumes half-width activations. Convert the routed rows once here,
    // rather than re-reading F32 and converting the same activation tile for every output-N tile.
    q.parallel_for(sycl::range<1>(elements), [=](sycl::id<1> id) {
        const size_t i = (size_t) id[0];
        packed[i] = (T) src1[i];
    });

    // Batch at most 16 active experts per launch. The row offsets are absolute in the already
    // expert-sorted activation/output buffers; zero-count experts can be skipped without gaps.
    const bool q5 = src0_type == GGML_TYPE_Q5_K;
    if constexpr (LOCAL_REDUCE) {
        for (const local_reduce_batch & batch : local_batches) {
            group gp = batch.experts;
            const int64_t group_rows = (int64_t) gp.off[gp.n] - gp.off[0];
            const int64_t avg_rows = group_rows / gp.n;
            if (avg_rows < 96) {
                const int ntiles = fill_tiles<small_config>(gp);
                if (q5) launch_layout<small_config, T, true, SUMMED, true>(q, gp, packed, group_products.get(), blocks, ntiles,
                    (int) M, (int) K, reordered, row_mapping, route_weights, n_ids, sum_dst, batch.row_begin);
                else    launch_layout<small_config, T, false, SUMMED, true>(q, gp, packed, group_products.get(), blocks, ntiles,
                    (int) M, (int) K, reordered, row_mapping, route_weights, n_ids, sum_dst, batch.row_begin);
            } else {
                const int ntiles = fill_tiles<large_config>(gp);
                if (q5) launch_layout<large_config, T, true, SUMMED, true>(q, gp, packed, group_products.get(), blocks, ntiles,
                    (int) M, (int) K, reordered, row_mapping, route_weights, n_ids, sum_dst, batch.row_begin);
                else    launch_layout<large_config, T, false, SUMMED, true>(q, gp, packed, group_products.get(), blocks, ntiles,
                    (int) M, (int) K, reordered, row_mapping, route_weights, n_ids, sum_dst, batch.row_begin);
            }
            // One work-item owns a final output element and reduces this batch's route slots
            // in registers. No global floating-point atomic is used in the local-reduction mode.
            reduce_group_partials(q, group_products.get(), route_to_compact_dev.get(),
                                  group_tokens_dev.get() + batch.token_begin, batch.token_count,
                                  batch.row_begin, batch.row_end, route_weights, n_ids, (int) M, sum_dst);
        }
    } else {
        int64_t e = 0;
        while (e < n_as) {
            group gp{};
            while (e < n_as && gp.n < MAX_EXPERTS) {
                const int64_t first = expert_row_offsets[e];
                const int64_t last  = expert_row_offsets[e + 1];
                if (last > first) {
                    gp.w[gp.n] = (const uint8_t *) src0_base + (size_t) e * expert_stride;
                    gp.off[gp.n] = (int32_t) first;
                    gp.off[gp.n + 1] = (int32_t) last;
                    ++gp.n;
                }
                ++e;
            }
            if (gp.n == 0) continue;

            const int64_t group_rows = (int64_t) gp.off[gp.n] - gp.off[0];
            const int64_t avg_rows = group_rows / gp.n;
            if (avg_rows < 96) {
                const int ntiles = fill_tiles<small_config>(gp);
                if (q5) launch_layout<small_config, T, true, SUMMED>(q, gp, packed, dst, blocks, ntiles, (int) M, (int) K, reordered,
                    row_mapping, route_weights, n_ids, sum_dst);
                else    launch_layout<small_config, T, false, SUMMED>(q, gp, packed, dst, blocks, ntiles, (int) M, (int) K, reordered,
                    row_mapping, route_weights, n_ids, sum_dst);
            } else {
                const int ntiles = fill_tiles<large_config>(gp);
                if (q5) launch_layout<large_config, T, true, SUMMED>(q, gp, packed, dst, blocks, ntiles, (int) M, (int) K, reordered,
                    row_mapping, route_weights, n_ids, sum_dst);
                else    launch_layout<large_config, T, false, SUMMED>(q, gp, packed, dst, blocks, ntiles, (int) M, (int) K, reordered,
                    row_mapping, route_weights, n_ids, sum_dst);
            }
        }
    }
    return true;
}
#endif // GGML_SYCL_XMX_AOT_SG == 8

static bool enabled() {
    static const bool value = [] {
        const char * e = std::getenv("GGML_SYCL_MOE_XMXG_Q4K_Q5K");
#if defined(GGML_SYCL_MOE_XMXG_Q4K_Q5K_DEFAULT)
        constexpr bool default_enabled = true;
#else
        constexpr bool default_enabled = false;
#endif
        // The environment variable is an explicit override: unset uses the
        // build-time default, "0" disables, and a nonzero value enables.
        const bool value = e == nullptr ? default_enabled : std::atoi(e) != 0;
        GGML_LOG_INFO("%s: %d\n", "GGML_SYCL_MOE_XMXG_Q4K_Q5K", value ? 1 : 0);
        return value;
    }();
    return value;
}

static bool local_reduce_enabled() {
    static const bool value = [] {
        const char * e = std::getenv("GGML_SYCL_MOE_DOWN_SUM_LOCAL");
        const bool value = e != nullptr && std::atoi(e) != 0;
        GGML_LOG_INFO("%s: %d\n", "GGML_SYCL_MOE_DOWN_SUM_LOCAL", value ? 1 : 0);
        return value;
    }();
    return value;
}
} // namespace fg_xmxg

bool ggml_sycl_grouped_xmxg_q4k_q5k(ggml_type src0_type, bool reordered, const void * src0_base,
                                     size_t expert_stride, const float * src1, int32_t src1_prec, float * dst,
                                     const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                     int64_t total_rows, ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    if (!fg_xmxg::enabled() || (src0_type != GGML_TYPE_Q4_K && src0_type != GGML_TYPE_Q5_K) ||
        g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (src0_base == nullptr || src1 == nullptr || dst == nullptr || expert_row_offsets == nullptr ||
        n_as <= 0 || n_as > INT32_MAX || M <= 0 || K <= 0 || M % 128 != 0 || K % QK_K != 0 ||
        M > INT32_MAX || K > INT32_MAX || total_rows <= 0 || total_rows > INT32_MAX || expert_stride == 0 || !stream) {
        return false;
    }
#if defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 8
    return false;
#else
    // This kernel uses the engine's 8-lane 8x8x16 XMX tile. Support both F16 (combo 4) and
    // BF16 (combo 7), respecting the same precision request as the existing fused-GEMM path.
    const int combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    if (combo != 4 && combo != 7) return false;

    const size_t block_bytes = src0_type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q5_K);
    if (expert_stride % block_bytes != 0 || expert_stride / block_bytes > INT32_MAX) return false;
    const int blocks = (int) (expert_stride / block_bytes);
    const uint64_t expected_blocks = (uint64_t) M * (uint64_t) (K / QK_K);
    if ((uint64_t) blocks != expected_blocks) return false;
    if (expert_row_offsets[0] != 0) return false;
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        if (expert_row_offsets[e + 1] < expert_row_offsets[e]) return false;
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
        if (expert_row_offsets[e + 1] > INT32_MAX) return false;
    }
    const int64_t valid_rows = expert_row_offsets[n_as];
    if (n_active == 0 || valid_rows <= n_active * 64 || valid_rows > total_rows) return false;

    if (combo == 4) {
        return fg_xmxg::run_grouped<sycl::half, false>(src0_type, reordered, src0_base, expert_stride,
                                                src1, dst, expert_row_offsets, n_as, M, K,
                                                valid_rows, blocks, pool, stream, nullptr, nullptr, 0, 0, nullptr);
    }
    using bf16 = sycl::ext::oneapi::bfloat16;
    return fg_xmxg::run_grouped<bf16, false>(src0_type, reordered, src0_base, expert_stride,
                                      src1, dst, expert_row_offsets, n_as, M, K,
                                      valid_rows, blocks, pool, stream, nullptr, nullptr, 0, 0, nullptr);
#endif
}

// Fused down-projection variant: route-weight each XMX accumulator tile and reduce it directly
// into [n_embd, n_tokens], avoiding the intermediate [n_embd, top_k, n_tokens] product tensor.
bool ggml_sycl_grouped_xmxg_q4k_q5k_sum(ggml_type src0_type, bool reordered, const void * src0_base,
                                         size_t expert_stride, const float * src1, int32_t src1_prec,
                                         const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                         int64_t total_rows, const mmid_row_mapping * row_mapping,
                                         const mmid_row_mapping * row_mapping_host,
                                         const float * route_weights, int n_ids, int64_t n_tokens,
                                         float * sum_dst, ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    if (!fg_xmxg::enabled() || (src0_type != GGML_TYPE_Q4_K && src0_type != GGML_TYPE_Q5_K) ||
        g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type)) {
        return false;
    }
    if (src0_base == nullptr || src1 == nullptr || expert_row_offsets == nullptr || row_mapping == nullptr ||
        route_weights == nullptr || sum_dst == nullptr || n_ids <= 0 || n_tokens <= 0 ||
        n_as <= 0 || n_as > INT32_MAX || M <= 0 || K <= 0 || M % 128 != 0 || K % QK_K != 0 ||
        M > INT32_MAX || K > INT32_MAX || total_rows <= 0 || total_rows > INT32_MAX ||
        (uint64_t) total_rows != (uint64_t) n_ids * (uint64_t) n_tokens ||
        (uint64_t) M > SIZE_MAX / sizeof(float) / (uint64_t) n_tokens || expert_stride == 0 || !stream) {
        return false;
    }
#if defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 8
    return false;
#else
    const int combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    if (combo != 4 && combo != 7) return false;

    const size_t block_bytes = src0_type == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q5_K);
    if (expert_stride % block_bytes != 0 || expert_stride / block_bytes > INT32_MAX) return false;
    const int blocks = (int) (expert_stride / block_bytes);
    const uint64_t expected_blocks = (uint64_t) M * (uint64_t) (K / QK_K);
    if ((uint64_t) blocks != expected_blocks || expert_row_offsets[0] != 0) return false;
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        if (expert_row_offsets[e + 1] < expert_row_offsets[e]) return false;
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
        if (expert_row_offsets[e + 1] > INT32_MAX) return false;
    }
    const int64_t valid_rows = expert_row_offsets[n_as];
    if (n_active == 0 || valid_rows <= n_active * 64 || valid_rows > total_rows) return false;

    const bool use_local_reduce = fg_xmxg::local_reduce_enabled();
    if (use_local_reduce && row_mapping_host == nullptr) {
        GGML_LOG_WARN("SYCL MoE down sum: host route mapping unavailable; using atomic reduction\n");
    }
    if (combo == 4) {
        if (use_local_reduce && row_mapping_host != nullptr && fg_xmxg::run_grouped<sycl::half, true, true>(src0_type, reordered, src0_base, expert_stride,
            src1, nullptr, expert_row_offsets, n_as, M, K, valid_rows, blocks, pool, stream,
            row_mapping, route_weights, n_ids, n_tokens, sum_dst, row_mapping_host)) {
            if (g_ggml_sycl_debug) {
                GGML_LOG_INFO("SYCL MoE down sum: grouped partial local-reduction path active\n");
            }
            return true;
        }
        if (use_local_reduce) {
            GGML_LOG_WARN("SYCL MoE down sum: local reduction unavailable; using atomic reduction\n");
        }
        return fg_xmxg::run_grouped<sycl::half, true>(src0_type, reordered, src0_base, expert_stride,
            src1, nullptr, expert_row_offsets, n_as, M, K, valid_rows, blocks, pool, stream,
            row_mapping, route_weights, n_ids, n_tokens, sum_dst);
    }
    using bf16 = sycl::ext::oneapi::bfloat16;
    if (use_local_reduce && row_mapping_host != nullptr && fg_xmxg::run_grouped<bf16, true, true>(src0_type, reordered, src0_base, expert_stride,
        src1, nullptr, expert_row_offsets, n_as, M, K, valid_rows, blocks, pool, stream,
        row_mapping, route_weights, n_ids, n_tokens, sum_dst, row_mapping_host)) {
        if (g_ggml_sycl_debug) {
            GGML_LOG_INFO("SYCL MoE down sum: grouped partial local-reduction path active\n");
        }
        return true;
    }
    if (use_local_reduce) {
        GGML_LOG_WARN("SYCL MoE down sum: local reduction unavailable; using atomic reduction\n");
    }
    return fg_xmxg::run_grouped<bf16, true>(src0_type, reordered, src0_base, expert_stride,
        src1, nullptr, expert_row_offsets, n_as, M, K, valid_rows, blocks, pool, stream,
        row_mapping, route_weights, n_ids, n_tokens, sum_dst);
#endif
}

bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                                    const float * src1, int32_t src1_prec, float * dst,
                                    const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t total_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream) {
    int64_t n_active = 0;
    for (int64_t e = 0; e < n_as; ++e) {
        n_active += expert_row_offsets[e + 1] > expert_row_offsets[e];
    }
    if (g_ggml_sycl_dynamic_precision == GGML_SYCL_DYNAMIC_PRECISION_F32 ||
        !ggml_sycl_xmx_gather_type_enabled(src0_type) || !fg_visit_type(src0_type, reordered, [](auto) {})) {
        return false;
    }
    if (!ggml_sycl_grouped_dequant_gemm_shape_ok(src0_type, M, K, total_rows, n_active)) {
        return false;
    }
    const int combo = fg_pick_combo(stream, GGML_TYPE_F32, src1_prec);
    if (combo < 0) {
        return false;
    }
    bool launched = false;
    fg_visit_combo(combo, [&](auto s) {
        launched = fg_grouped_run<decltype(s)>(src0_type, reordered, src0_base, expert_stride, src1, dst,
                                               expert_row_offsets, n_as, M, K, tiles, pool, stream);
    });
    return launched;
}
