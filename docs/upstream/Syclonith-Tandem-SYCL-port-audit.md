# Syclonith × Tandem MoE cache: SYCL port audit

## Scope inspected

This audit is based on the uploaded Syclonith source archive, the six Tandem reference files, and `tandem.patch`.

Syclonith baseline reported by the user: `0b77cf229d6bdd030efb173fdbc850114ed976c9`.

Tandem patch baseline: `f5222f25c`.

The Tandem patch applies cleanly to its intended stock llama.cpp baseline, but does not apply cleanly to Syclonith. The user's clean working tree means there are no local modifications to account for.

## Findings

### 1. Syclonith already has a CUDA-specific MoE cache provider

Relevant files:

- `ggml/src/ggml-backend-moe-cache.h`
- `ggml/src/ggml-backend.cpp`
- `ggml/src/ggml-cuda/moe-cache.cu`
- `ggml/src/ggml-cuda/moe-cache.cuh`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `src/llama-context.cpp`

The provider is exposed as one global function table:

```cpp
struct ggml_moe_cache_api ggml_moe_cache;
```

CUDA registers itself from `ggml_backend_cuda_reg()` via `ggml_moe_cache_register(&reg)`. Registration refuses to replace a different existing owner. The scheduler and CPU backend consume this global provider table.

The existing provider has advanced CUDA-specific functionality: hosted MoE block execution, cache management, CPU expert execution, prefetch/ring support, compute-buffer borrowing, KV lending and related scheduling hooks.

### 2. SYCL has a separate `MUL_MAT_ID` implementation

Relevant file:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`

`ggml_sycl_mul_mat_id()` already supports single-token fused MMVQ and multi-token expert grouping / grouped GEMM. It reads the source expert tensor directly and does not currently consult `ggml_moe_cache`.

The current function also explicitly rejects SYCL split buffers for this operation. The proposed first port should not introduce tensor-parallel or split-buffer assumptions.

### 3. Tandem's cache is a graph-level cache, not a drop-in GGML provider

Relevant files:

- `src/llama-moe-cache.cpp`
- `src/llama-moe-cache.h`

Tandem's cache is integrated through model graph construction and the scheduler copy callback. Its cache stores expert banks on backend devices, maintains host slot maps, and rewrites routed expert IDs to cache slots. The CPU hybrid path computes selected misses and adds their contribution to the GPU result.

The cache constructor already accepts both `GGML_BACKEND_DEVICE_TYPE_GPU` and `GGML_BACKEND_DEVICE_TYPE_IGPU`, and most core movement operations use generic backend APIs such as `ggml_backend_tensor_set_async`, `ggml_backend_tensor_get_async`, events and backend synchronization.

That is promising for SYCL portability, but it is not proof of compatibility. There are CUDA-specific assumptions elsewhere in the implementation (for example MMQ padding, GPU-fetch optimisations, expert chunk probing and some fast-path choices). These must be disabled or adapted for SYCL.

### 4. Why a direct registration patch would be wrong

Adding a SYCL call to `ggml_moe_cache_register()` would not implement Tandem's graph-level cache. It would also compete with the existing single-owner CUDA provider. The current provider API's hosted execution callbacks are consumed by CUDA-specific paths, so a SYCL backend must not be registered as if it implemented those CUDA callbacks.

The safe integration is additive at the llama graph/cache layer: keep the existing Syclonith CUDA provider untouched, and enable a separate Tandem-style cache path only for layers assigned to supported SYCL backends.

## Recommended staged port

### Stage A — Basic SYCL expert-slot cache (first code milestone)

Include only:

- backend selection for GPU and Intel integrated GPU devices;
- per-device expert banks and host slot maps;
- generic backend tensor transfers;
- LRU admission/eviction and slot-map updates;
- graph construction that substitutes cache-bank expert tensors and slot IDs;
- no CUDA-only MMQ padding, GPU fetch, lookahead, doorbell, ring, expert chunks or CUDA graph assumptions;
- no hybrid CPU computation yet.

Correctness gates:

1. Cache-off output matches the current Syclonith baseline.
2. Cache-on output matches cache-off output within the existing numerical tolerance.
3. Cache hits are verified against source expert bytes.
4. Eviction/reload cycles preserve output correctness.
5. The CUDA build and CUDA cache path remain unchanged.

### Stage B — Hybrid CPU execution

Only after Stage A is correct:

- enable Tandem's CPU expert path for supported activation/weight combinations;
- initially use the conservative synchronous host staging path;
- keep CPU and GPU row ownership disjoint;
- verify every routed row is computed exactly once;
- add overlap / pinned-memory signalling only after correctness.

### Stage C — Performance features

Evaluate independently, not as part of the initial port:

- asynchronous copy streams/events;
- lookahead prefetch;
- late expert uploads;
- large-batch expert chunking;
- GPU fetch / doorbell;
- graph capture / replay interactions.

## Current status and limitations

- Source archives extracted and inspected.
- Tandem patch's target baseline identified.
- Existing Syclonith CUDA provider and SYCL `MUL_MAT_ID` paths identified.
- Tandem's generic backend operations and CUDA-specific assumptions identified.
- No Syclonith integration patch has been applied or generated yet.
- No compilation or runtime benchmark has been performed. The uploaded source is a source archive, not a configured Rocky Linux / oneAPI build environment.

A safe implementation patch must be produced and reviewed in the staged order above. A partial `git apply --reject` result is not a valid integration and must not be used.
