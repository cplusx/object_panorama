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
- `pytorch-lightning`
- `deepspeed` (only when using Lightning DeepSpeed strategies)

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
- `training/`: objective builders, optimizer/scheduler setup, the smoke/debug trainer, and Lightning formal-training wrappers.
- `evaluation/`: tensor dumps and preview image generation for debug runs.
- `tools/inspect_manifest_batch.py`: inspect one batch from a manifest-driven config.
- `tools/overfit_rectangular_conditional_jit.py`: tiny overfit smoke entrypoint.
- `tools/train_lightning_rectangular_conditional_jit.py`: formal training entrypoint via PyTorch Lightning.

Current real-data tensor semantics are documented in `docs/edge3d_training_data_logic.md`.

### Debug

Inspect one batch:

```bash
python tools/inspect_manifest_batch.py configs/experiment/exp_b32_sparse_debug.yaml
```

Run a tiny overfit smoke:

```bash
python tools/overfit_rectangular_conditional_jit.py configs/experiment/exp_b32_sparse_debug.yaml --device cuda
```

The following files remain debug-only and are not the formal training path:

- `training/trainer.py`
- `tools/inspect_manifest_batch.py`
- `tools/overfit_rectangular_conditional_jit.py`

### Manifest Format

Each JSONL row must contain:

```json
{
	"sample_id": "abc_0001",
	"tensor_path": "relative/or/absolute/path/to/sample.npz",
	"meta": {
		"anything": "optional"
	}
}
```

For the modality-native training path, the dataloader reads one `.npz` sample and returns:

- `model_rgb`
- `model_depth`
- `model_normal`
- `edge_depth`

The training module then builds the actual tensors used by the model:

- `target = edge_depth` with `3` hits
- `condition = concat(model_depth, model_normal)`
- `sample = (1 - t) * target + t * noise`

Current default condition channel count is `20`:

- `5` from `model_depth`
- `15` from `model_normal`

Legacy `condition_type_id` support still exists in the older triplet-manifest code path, but the current debug and real training configs now use modality manifests instead.

### Formal Training

Lightning single-GPU training:

```bash
python tools/train_lightning_rectangular_conditional_jit.py \
	configs/experiment/exp_b32_sparse_train.yaml \
	--device cuda \
	--precision 32-true \
	--strategy auto
```

Lightning resume:

```bash
python tools/train_lightning_rectangular_conditional_jit.py \
	configs/experiment/exp_b32_sparse_train.yaml \
	--resume runs/exp_b32_sparse_train/checkpoints/last.ckpt
```

Lightning + DeepSpeed:

```bash
python tools/train_lightning_rectangular_conditional_jit.py \
	configs/experiment/exp_b32_sparse_train.yaml \
	--device cuda \
	--precision 32-true \
	--strategy deepspeed_stage_2
```

Lightning handles fp32 training, resume, grad accumulation, checkpointing, and distributed strategy management. The repository-local simple trainer stays in place only for smoke/debug use.