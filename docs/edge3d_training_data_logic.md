# Current Edge3D Training Data Logic

## Scope

This note documents the current training semantics after the modality-native refactor.

The dataloader no longer reads pre-derived `input.npy`, `condition.npy`, and `target.npy` tensors.
It now reads raw per-sample modality tensors and lets the training code derive model input, condition, and supervision on the fly.

## 1. What The Dataloader Returns

The modality dataloader returns one sample as:

- `model_rgb`
- `model_depth`
- `model_normal`
- `edge_depth`

Those come from one mixed-precision `.npz` produced by `edge3d_export_training_tensors.py` or `edge3d_export_training_tensors_parallel.py`.

For the default exporter layout:

- `model_rgb`: `5` hits x `3` channels -> flattened to `15` channels
- `model_depth`: `5` hits -> `5` channels
- `model_normal`: `5` hits x `3` channels -> flattened to `15` channels
- `edge_depth`: `3` hits -> `3` channels

So the dataloader emits CHW tensors with shapes:

- `model_rgb`: `[15, H, W]`
- `model_depth`: `[5, H, W]`
- `model_normal`: `[15, H, W]`
- `edge_depth`: `[3, H, W]`

No training-role assignment happens inside the dataloader.

## 2. Condition Channel Count

The training condition is built by concatenating:

- `model_rgb`
- `model_depth`
- `model_normal`

So the current condition channel count is:

$$15 + 5 + 15 = 35$$

That is why the model configs use:

- `condition_channels_per_type: [35, 35, 35]`

The current training path uses a single fixed condition layout, but the model still keeps the existing three-stem interface. The training module therefore supplies a constant `condition_type_id = 0` internally.

## 3. What The Training Module Builds

`RectangularConditionalJiTLightningModule` no longer uses the old `paired_supervised` path.

It now builds the training batch with one fixed flow-matching style rule:

- `target = edge_depth`
- `condition = concat(model_rgb, model_depth, model_normal)`
- `noise ~ N(0, I)`
- `t ~ Uniform(t_min, t_max)`
- `sample = x_t = (1 - t) * target + t * noise`

So the actual model input is the noised `edge_depth` target, not a separately stored `input.npy`.

## 4. Supervision Semantics

The model is currently supervised to predict the `3`-hit edge-depth tensor.

That means:

- training target is `edge_depth`
- output channel count must be `3`
- input channel count must also be `3`, because the input is the noisy version of that same target

This is why the current model configs now use:

- `image_in_channels: 3`
- `image_out_channels: 3`

## 5. Manifest Format

The current modality manifest format is minimal. Each row needs:

```json
{
	"sample_id": "edge3d_overfit_0000",
	"tensor_path": "data/samples/edge3d_overfit_0000.npz",
	"meta": {
		"split": "train"
	}
}
```

The JSONL manifest does not store `input_path`, `condition_path`, or `target_path` anymore for the new training path.

## 6. Current Debug / Overfit Sample

The workspace debug path now uses exactly one corrected sample:

- `data/samples/edge3d_overfit_0000.npz`

That sample was copied from the canonical-edge export under:

- `analysis/edge3d_canonical_edge_100/training_tensors_gpu/00001ec0d78549e1b8c2083a06105c29.npz`

The debug manifest is:

- `data/train_manifest.jsonl`

And the debug data config is:

- `configs/data/manifest_example.yaml`

## 7. Current Real Subset Path

The current `20 / 4` real subset configs were also converted to modality manifests.

They now read `.npz` tensors directly from:

- `analysis/edge3d_canonical_edge_100/training_tensors_gpu`

through:

- `data/manifest_real_train.jsonl`
- `data/manifest_real_val.jsonl`

So the real subset and the debug overfit path now use the same modality-native training semantics.

## 8. Bottom Line

The current training stack is now:

1. Dataloader returns raw modalities: `model_rgb`, `model_depth`, `model_normal`, `edge_depth`.
2. Training module builds `condition = concat(model_rgb, model_depth, model_normal)`.
3. Training target is `edge_depth` with `3` hits.
4. Model input is a noisy version of that same `edge_depth`, using the flow-matching-style linear interpolation with Gaussian noise.

That is the current source of truth for training semantics in this repository.