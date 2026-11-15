# GPU-first SYCL MoE decode tuning

The default GPU-resident path has a reordered Q6_K MoE GEMV kernel that loads an
activation fragment once and applies it to multiple output rows. The current
default is two rows per subgroup. To benchmark alternative occupancy/activation
reuse trade-offs, this branch adds an opt-in runtime setting:

```bash
GGML_SYCL_MOE_Q6_ROWS_PER_SG=1   # one output row per subgroup
GGML_SYCL_MOE_Q6_ROWS_PER_SG=2   # existing default
GGML_SYCL_MOE_Q6_ROWS_PER_SG=4   # more activation reuse, potentially less occupancy
```

Unset or invalid values retain the existing default of `2`. This setting only
affects the reordered Q6_K single-token MoE decode kernel; it does not affect
prompt processing, other quantisations, or the host-resident Stage A cache. It
requires the normal SYCL weight reorder optimisation to be enabled and selected.

This is a tuning control, not a claimed performance improvement. Test all three
values with the same GPU-resident model, prompt, seed, context and generation
length. Compare repeated generation throughput and latency, and check output
correctness. A value of `4` may lose on integrated GPUs if the reduction in
subgroup parallelism or increased register use outweighs activation reuse.
