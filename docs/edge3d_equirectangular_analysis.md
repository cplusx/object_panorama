# Edge3D Equirectangular Export Analysis

## Scope

This note summarizes the current tensor export at resolution `512 x 1024` with `max_hits=5`.

- Dataset slice: first `1000` UIDs
- Successful exports: `998`
- Failed exports: `2`
- Current exporter backend in repo: `gpu_exact`
- The slice statistics below are retained as historical diagnostics; use the current `gpu_exact` backend for any new runs.
- Output tensors:
  - `model_tensor`: `(35, 512, 1024)`
  - `edge_tensor`: `(5, 512, 1024)`

Machine-readable outputs:

- `/home/devdata/edge3d_data/equirectangular_data/analysis_first1000_fp16_and_hits.json`
- `/home/devdata/edge3d_data/equirectangular_data/analysis_first1000_quantization.json`
- `/home/devdata/edge3d_data/equirectangular_data/export_summary.json`

## Export Failures

Two UIDs failed because the downloaded geometry resolved to `Path3D` instead of `Trimesh`:

- `0035230dbe124d64a67ffedefea75323`
- `00bb4a535959499685ff1ec5339b1be4`

Those failures are now skipped instead of aborting the whole batch.

## Edge Multi-Hit Retention

The edge tensor stores one depth layer per hit. Measured against the full `5-hit` representation, the cumulative retained valid edge entries are:

| Kept layers | Retained vs 5-hit |
| --- | ---: |
| `1 hit` | `80.31%` |
| `2 hit` | `94.31%` |
| `3 hit` | `97.96%` |
| `4 hit` | `99.38%` |
| `5 hit` | `100.00%` |

Interpretation:

- `1 hit` already keeps most edge signal, but drops about `19.69%` of valid edge-depth entries.
- `2 hit` is a strong compression point if we want to preserve most self-occlusion structure.
- `4 hit` is almost lossless relative to `5 hit`.

## Quantization Error

The numbers below are roundtrip error after `fp32 -> target_dtype -> fp32`, measured only on valid entries for each modality.

### FP16

| Modality | MAE | RMSE | Max abs | Relative MAE |
| --- | ---: | ---: | ---: | ---: |
| `model_rgb` | `5.65e-05` | `6.99e-05` | `4.88e-04` | `0.0165%` |
| `model_depth` | `4.33e-04` | `2.13e-03` | `6.25e-02` | `0.0176%` |
| `model_normal` | `3.82e-05` | `7.31e-05` | `2.44e-04` | `0.0096%` |
| `edge_depth` | `1.04e-04` | `1.40e-04` | `4.88e-04` | `0.0175%` |

FP16 is effectively safe for all four modalities in this slice.

### FP8 E4M3FN

| Modality | MAE | RMSE | Max abs | Relative MAE |
| --- | ---: | ---: | ---: | ---: |
| `model_rgb` | `6.72e-03` | `8.53e-03` | `6.25e-02` | `1.97%` |
| `model_depth` | `5.51e-02` | `2.70e-01` | `8.00e+00` | `2.25%` |
| `model_normal` | `4.57e-03` | `8.97e-03` | `3.12e-02` | `1.15%` |
| `edge_depth` | `1.32e-02` | `1.77e-02` | `6.25e-02` | `2.21%` |

### FP8 E5M2

| Modality | MAE | RMSE | Max abs | Relative MAE |
| --- | ---: | ---: | ---: | ---: |
| `model_rgb` | `1.31e-02` | `1.68e-02` | `1.00e-01` | `3.83%` |
| `model_depth` | `1.08e-01` | `5.24e-01` | `1.60e+01` | `4.41%` |
| `model_normal` | `8.86e-03` | `1.76e-02` | `6.25e-02` | `2.23%` |
| `edge_depth` | `2.60e-02` | `3.48e-02` | `1.25e-01` | `4.35%` |

### Precision Recommendation

- Keep `model_depth` and `edge_depth` at least in `fp16`. Their FP8 worst-case errors are large enough to be risky for geometry-sensitive supervision.
- `model_normal` is the most FP8-tolerant modality. If storage pressure is strong, `float8_e4m3fn` is the best low-precision candidate.
- `model_rgb` is more tolerant than the depth channels, but still incurs visible-scale quantization compared with `fp16`.
- Between the two tested FP8 formats, `float8_e4m3fn` is consistently better than `float8_e5m2` on this dataset slice.

## Profiling Results

The `100`-sample end-to-end profiling run includes download time by forcing a fresh Objaverse cache for each configuration.

- `/home/devdata/edge3d_data/equirectangular_profile_runs/profile_report.json`

All three runs completed successfully with `100 / 100` exports.

| Configuration | Elapsed sec | Sec per sample | Throughput | Speedup vs 1GPUx1 |
| --- | ---: | ---: | ---: | ---: |
| `1 GPU x 1 worker` | `754.63` | `7.55` | `0.133 samples/s` | `1.00x` |
| `2 GPU x 1 worker each` | `389.59` | `3.90` | `0.257 samples/s` | `1.94x` |
| `2 GPU x 2 workers each` | `212.32` | `2.12` | `0.471 samples/s` | `3.55x` |

Interpretation:

- Moving from `1 GPU x 1 worker` to `2 GPU x 1 worker each` is close to ideal two-GPU scaling.
- Moving from `2 GPU x 1 worker each` to `2 GPU x 2 workers each` gives another `1.83x` speedup, so the extra worker per GPU is still useful in the download-inclusive setting.
- Compared with the single-worker baseline, `2 GPU x 2 workers each` improves throughput by about `3.55x` and reduces average end-to-end time from `7.55 s` per sample to `2.12 s` per sample.