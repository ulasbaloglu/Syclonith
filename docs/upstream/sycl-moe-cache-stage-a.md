# Stage A: SYCL MoE expert-slot cache

Patch baseline: Syclonith commit `0b77cf229d6bdd030efb173fdbc850114ed976c9`.

Apply from the repository root after creating a feature branch:

```bash
git switch -c sycl-moe-cache-stage-a
git apply --check /path/to/Syclonith-Tandem-Stage-A.patch
git apply /path/to/Syclonith-Tandem-Stage-A.patch
```

Stage A adds an opt-in, cache-only expert bank for MoE layers assigned to the
SYCL backend. It is intentionally independent of the existing CUDA MoE cache
provider and does not replace the existing SYCL `MUL_MAT_ID`/grouped-GEMM
kernels.

## Enable

Set the cache budget in MiB before starting the server:

```bash
GGML_SYCL_MOE_CACHE_MB=2048 \
  ./build-sycl/bin/llama-server ...
```

The feature is disabled when `GGML_SYCL_MOE_CACHE_MB` is unset or zero. The
budget is a hard upper bound on the logical expert-bank payload; backend buffer
alignment can add a small amount of allocation overhead.

For the first test, avoid `--no-host`: Stage A currently requires the original
expert tensors to be host-resident weight buffers. If no eligible layers are
found, startup logs that the cache was disabled.

## What is cached

- Layers assigned to a backend whose registry name is `SYCL`.
- Expert tensors in `ffn_gate_up_exps`, `ffn_gate_exps`, `ffn_up_exps`, and
  `ffn_down_exps`.
- Only layers whose expert tensors are contiguous and host-resident weights.
- One slot map and a set of device banks per eligible layer.

The requested budget is apportioned across eligible layers in proportion to
their expert-weight footprint. If all experts of a layer fit, that layer is
fully uploaded once at initialization. Otherwise, the layer uses an LRU bank
filled on demand.

## Execution behaviour and limitations

- The graph maps routed expert IDs to cache slots and runs the existing SYCL
  `MUL_MAT_ID` implementation against the bank tensors.
- Partial-cache fills happen in the scheduler's input-copy callback. The
  callback reads routed IDs from the preceding split, uploads missing experts,
  and then uploads the updated slot map on the same backend queue.
- Partial-cache graphs are used only for ubatches of at most 32 tokens and only
  when the configured slot count can hold the worst-case `tokens * experts`
  selection. Other shapes use the original full-weight path.
- Fully resident layers can use the bank for larger ubatches.
- CPU execution of cache misses, asynchronous prefetch, speculative-decoding
  integration, and cache sharing between contexts are not part of Stage A.
- Partial-cache operation includes a device-to-host readback of routed IDs;
  this is a correctness-first implementation and is not guaranteed to improve
  throughput. Measure it against the same workload with the variable unset.
- Pipeline-parallel scheduling is not enabled for this cache implementation.
- The SYCL expert-bank tensors deliberately skip the existing in-place expert
  reorder optimisation. Reordering a bank once would make later LRU slot fills
  (which copy source-layout bytes) invalid. This preserves correctness at the
  possible cost of some decode performance for reorderable quantisations.

## Expected startup and shutdown diagnostics

When enabled, startup reports the number of eligible layers and the logical
bank allocation. On context shutdown, the cache reports hit/miss/upload
counters for partial-cache admissions.

## Suggested validation

1. Run once with `GGML_SYCL_MOE_CACHE_MB` unset and save output for a fixed
   prompt and generation seed.
2. Run the same model, prompt, seed, context, batch and server settings with a
   modest cache budget (for example, `GGML_SYCL_MOE_CACHE_MB=1024`).
3. Confirm startup reports at least one cached SYCL layer.
4. Compare generated tokens or logits for correctness before comparing speed.
5. Repeat the same prompt several times to distinguish cold admissions from
   cache hits. Compare prompt throughput, generation throughput and total
   latency; do not infer success from hit rate alone.

Stage A is an experimental first port. It has been syntax-compiled for the
shared C++ sources in a CPU-only build configuration, but has not yet been
compiled with the Intel oneAPI/SYCL toolchain or run on an Intel GPU.
