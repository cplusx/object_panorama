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
- This stage is inference-only. No training loop or 3D data integration is included in the JiT path yet.