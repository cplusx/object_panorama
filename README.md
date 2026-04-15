# object_panorama

## JiT Bring-Up

This repository now includes a local Diffusers-native JiT implementation for inference-only bring-up.

### Environment

Use the `pytorch3d` conda environment for JiT work. The required runtime packages are:

- `torch`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- `einops`

### Files

- `models/jit_layers.py`: shared JiT layers and rotary helpers.
- `models/jit_model.py`: Diffusers-native JiT model presets.
- `models/jit_checkpoint_loader.py`: official checkpoint and EMA-aware loading helpers.
- `schedulers/scheduling_jit_flow.py`: JiT Euler and Heun sampling scheduler.
- `pipelines/pipeline_jit.py`: custom Diffusers JiT inference pipeline.
- `convert_jit_checkpoint.py`: convert an official public checkpoint into a local Diffusers bundle.
- `sample_jit.py`: run class-conditional sampling through the local JiT pipeline.

### Convert An Official Checkpoint

```bash
python convert_jit_checkpoint.py \
	--checkpoint pretrained_weights/checkpoint-last.pth \
	--output-dir pretrained_weights/jit_l16_256 \
	--variant ema1
```

This writes:

- `pretrained_weights/jit_l16_256/transformer`
- `pretrained_weights/jit_l16_256/scheduler`
- `pretrained_weights/jit_l16_256/bundle_report.json`

### Sample Images

From a converted bundle:

```bash
python sample_jit.py \
	--bundle-dir pretrained_weights/jit_l16_256 \
	--class-labels 0,1,2,3 \
	--num-inference-steps 50 \
	--guidance-scale 2.4 \
	--sampling-method heun \
	--seed 0 \
	--output-dir jit_samples/jit_l16_demo
```

Or directly from an official public checkpoint:

```bash
python sample_jit.py \
	--checkpoint pretrained_weights/checkpoint-last.pth \
	--variant ema1 \
	--class-labels 207,207,207,207 \
	--num-inference-steps 50 \
	--guidance-scale 2.4 \
	--sampling-method heun \
	--seed 0 \
	--output-dir jit_samples/from_public_ckpt
```

### Notes

- Public JiT uses `model_ema1` for generation by default. The local tools default to `--variant ema1` for that reason.
- JiT predicts the clean sample, not DDPM epsilon noise. The scheduler in `schedulers/scheduling_jit_flow.py` keeps that sampling logic explicit.
- The class-conditional public-checkpoint sampling path remains inference-focused. The rectangular conditional training stack lives separately under `models/conditional_jit/`, `datasets/`, `training/`, and `tools/`.

## Rectangular Conditional JiT Training

The repository now includes a first-pass training stack for the rectangular conditional JiT model.

### Training Files

- `models/conditional_jit/`: split implementation for geometry helpers, condition tower, adapters, and the main rectangular conditional JiT model.
- `datasets/`: JSONL manifest dataset, tensor loading, transforms, and collate logic.
- `training/`: objective builders, optimizer/scheduler setup, train/eval steps, checkpointing, and the simple trainer.
- `evaluation/`: tensor dumps and preview image generation for debug runs.
- `tools/inspect_manifest_batch.py`: inspect one batch from a manifest-driven config.
- `tools/overfit_rectangular_conditional_jit.py`: overfit the first 8 samples of the train split.
- `tools/train_rectangular_conditional_jit.py`: launch normal single-GPU training from an experiment config.

### Manifest Format

Each JSONL row must contain:

```json
{
	"sample_id": "abc_0001",
	"input_path": "relative/or/absolute/path/to/input.npy",
	"condition_path": "relative/or/absolute/path/to/condition.npy",
	"target_path": "relative/or/absolute/path/to/target.npy",
	"condition_type_id": 0,
	"meta": {
		"anything": "optional"
	}
}
```

### Training Entry Points

Inspect one batch:

```bash
python tools/inspect_manifest_batch.py configs/experiment/exp_b32_sparse_debug.yaml
```

Overfit the first 8 samples:

```bash
python tools/overfit_rectangular_conditional_jit.py configs/experiment/exp_b32_sparse_debug.yaml --device cuda
```

Run normal training:

```bash
python tools/train_rectangular_conditional_jit.py configs/experiment/exp_b32_sparse_train.yaml --device cuda
```