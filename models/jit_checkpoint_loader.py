import json
from pathlib import Path
from typing import Any

import torch

from .jit_model import JiTModel, create_jit_model


PUBLIC_CHECKPOINT_VARIANTS = {
    "raw": "model",
    "model": "model",
    "ema1": "model_ema1",
    "model_ema1": "model_ema1",
    "ema2": "model_ema2",
    "model_ema2": "model_ema2",
}


def _get_checkpoint_arg(args: Any, name: str, default: Any = None) -> Any:
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _strip_prefix_from_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    prefix_length = len(prefix)
    return {
        key[prefix_length:]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _normalize_public_jit_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("net.") for key in state_dict):
        stripped = _strip_prefix_from_state_dict(state_dict, "net.")
        if stripped:
            return stripped
    return state_dict


def _match_preset_from_inferred_config(inferred: dict[str, Any]) -> str | None:
    candidates = []
    for preset_name, preset_config in create_jit_model.__globals__["JIT_PRESET_CONFIGS"].items():
        if inferred.get("depth") != preset_config["depth"]:
            continue
        if inferred.get("hidden_size") != preset_config["hidden_size"]:
            continue
        if inferred.get("patch_size") != preset_config["patch_size"]:
            continue
        if inferred.get("bottleneck_dim") != preset_config["bottleneck_dim"]:
            continue
        if inferred.get("in_context_len") != preset_config["in_context_len"]:
            continue
        candidates.append(preset_name)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _infer_architecture_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    normalized_state_dict = _normalize_public_jit_state_dict(state_dict)
    if not normalized_state_dict:
        return {}

    keys = normalized_state_dict.keys()
    block_indices = {
        int(key.split(".")[1])
        for key in keys
        if key.startswith("blocks.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
    }
    depth = len(block_indices)

    pos_embed = normalized_state_dict.get("pos_embed")
    proj1_weight = normalized_state_dict.get("x_embedder.proj1.weight")
    proj2_weight = normalized_state_dict.get("x_embedder.proj2.weight")
    label_embed = normalized_state_dict.get("y_embedder.embedding_table.weight")
    q_norm_weight = normalized_state_dict.get("blocks.0.attn.q_norm.weight")
    in_context_posemb = normalized_state_dict.get("in_context_posemb")

    inferred: dict[str, Any] = {}
    if depth > 0:
        inferred["depth"] = depth
    if proj2_weight is not None:
        inferred["hidden_size"] = int(proj2_weight.shape[0])
        inferred["bottleneck_dim"] = int(proj2_weight.shape[1])
    if proj1_weight is not None:
        inferred["patch_size"] = int(proj1_weight.shape[-1])
        inferred["in_channels"] = int(proj1_weight.shape[1])
    if pos_embed is not None and proj1_weight is not None:
        num_patches = int(pos_embed.shape[1])
        grid_size = int(round(num_patches**0.5))
        inferred["input_size"] = grid_size * int(proj1_weight.shape[-1])
    if label_embed is not None:
        inferred["num_classes"] = int(label_embed.shape[0] - 1)
    if q_norm_weight is not None and inferred.get("hidden_size") is not None:
        head_dim = int(q_norm_weight.shape[0])
        if head_dim > 0 and inferred["hidden_size"] % head_dim == 0:
            inferred["num_heads"] = inferred["hidden_size"] // head_dim
    if in_context_posemb is not None:
        inferred["in_context_len"] = int(in_context_posemb.shape[1])

    inferred["preset_name"] = _match_preset_from_inferred_config(inferred)
    if inferred["preset_name"] is not None:
        preset_config = create_jit_model.__globals__["JIT_PRESET_CONFIGS"][inferred["preset_name"]]
        inferred.setdefault("num_heads", preset_config["num_heads"])
        inferred.setdefault("in_context_start", preset_config["in_context_start"])
    return inferred


def load_public_jit_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dict, got {type(checkpoint)!r}")
    return checkpoint


def infer_jit_model_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args")
    inferred = {
        "preset_name": _get_checkpoint_arg(args, "model"),
        "input_size": int(_get_checkpoint_arg(args, "img_size", 256)) if args is not None else None,
        "num_classes": int(_get_checkpoint_arg(args, "class_num", 1000)) if args is not None else None,
        "attn_drop": float(_get_checkpoint_arg(args, "attn_dropout", 0.0)) if args is not None else None,
        "proj_drop": float(_get_checkpoint_arg(args, "proj_dropout", 0.0)) if args is not None else None,
        "t_eps": float(_get_checkpoint_arg(args, "t_eps", 5e-2)) if args is not None else None,
        "noise_scale": float(_get_checkpoint_arg(args, "noise_scale", 1.0)) if args is not None else None,
        "sampling_method": str(_get_checkpoint_arg(args, "sampling_method", "heun")) if args is not None else None,
        "num_sampling_steps": int(_get_checkpoint_arg(args, "num_sampling_steps", 50)) if args is not None else None,
        "cfg": float(_get_checkpoint_arg(args, "cfg", 1.0)) if args is not None else None,
        "interval_min": float(_get_checkpoint_arg(args, "interval_min", 0.0)) if args is not None else None,
        "interval_max": float(_get_checkpoint_arg(args, "interval_max", 1.0)) if args is not None else None,
    }
    inferred = {key: value for key, value in inferred.items() if value is not None}

    if inferred.get("preset_name") is None:
        try:
            inferred.update(
                {
                    key: value
                    for key, value in _infer_architecture_config_from_state_dict(
                        extract_public_jit_state_dict(checkpoint, variant="ema1")
                    ).items()
                    if value is not None and key not in inferred
                }
            )
        except Exception:
            pass
    return inferred


def extract_public_jit_state_dict(checkpoint: dict[str, Any], variant: str = "ema1") -> dict[str, torch.Tensor]:
    normalized_variant = variant.lower()
    key = PUBLIC_CHECKPOINT_VARIANTS.get(normalized_variant)
    if key is None:
        available = ", ".join(sorted(PUBLIC_CHECKPOINT_VARIANTS))
        raise ValueError(f"Unknown checkpoint variant '{variant}'. Available variants: {available}")

    if key in checkpoint:
        state_dict = checkpoint[key]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint entry '{key}' is not a state dict")
        return _normalize_public_jit_state_dict(state_dict)

    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return _normalize_public_jit_state_dict(checkpoint)

    available_keys = ", ".join(sorted(checkpoint.keys()))
    raise KeyError(f"Checkpoint is missing '{key}'. Available keys: {available_keys}")


def load_public_weights_into_model(
    model: JiTModel,
    checkpoint_or_path: dict[str, Any] | str | Path,
    variant: str = "ema1",
    strict: bool = False,
) -> dict[str, Any]:
    checkpoint = (
        checkpoint_or_path
        if isinstance(checkpoint_or_path, dict)
        else load_public_jit_checkpoint(checkpoint_or_path, map_location="cpu")
    )
    state_dict = extract_public_jit_state_dict(checkpoint, variant=variant)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    matched_keys = sorted(model_keys & checkpoint_keys)
    return {
        "variant": variant,
        "strict": strict,
        "matched_key_count": len(matched_keys),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "matched_keys": matched_keys,
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
    }


def instantiate_model_from_public_checkpoint(
    checkpoint_or_path: dict[str, Any] | str | Path,
    *,
    preset_name: str | None = None,
    input_size: int | None = None,
    num_classes: int | None = None,
    attn_drop: float | None = None,
    proj_drop: float | None = None,
) -> tuple[JiTModel, dict[str, Any], dict[str, Any]]:
    checkpoint = (
        checkpoint_or_path
        if isinstance(checkpoint_or_path, dict)
        else load_public_jit_checkpoint(checkpoint_or_path, map_location="cpu")
    )
    inferred = infer_jit_model_config_from_checkpoint(checkpoint)

    resolved_preset = preset_name or inferred.get("preset_name")
    resolved_input_size = int(input_size if input_size is not None else inferred.get("input_size", 256))
    resolved_num_classes = int(num_classes if num_classes is not None else inferred.get("num_classes", 1000))
    resolved_attn_drop = float(attn_drop if attn_drop is not None else inferred.get("attn_drop", 0.0))
    resolved_proj_drop = float(proj_drop if proj_drop is not None else inferred.get("proj_drop", 0.0))

    if resolved_preset is not None:
        model = create_jit_model(
            resolved_preset,
            input_size=resolved_input_size,
            num_classes=resolved_num_classes,
            attn_drop=resolved_attn_drop,
            proj_drop=resolved_proj_drop,
        )
        return model, checkpoint, inferred

    required_keys = ["patch_size", "hidden_size", "depth", "num_heads", "bottleneck_dim", "in_context_len"]
    missing_required = [key for key in required_keys if key not in inferred]
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise ValueError(
            "Unable to infer JiT architecture from checkpoint; "
            f"missing derived fields: {missing_text}. Pass preset_name explicitly"
        )

    model = JiTModel(
        input_size=resolved_input_size,
        patch_size=int(inferred["patch_size"]),
        in_channels=int(inferred.get("in_channels", 3)),
        hidden_size=int(inferred["hidden_size"]),
        depth=int(inferred["depth"]),
        num_heads=int(inferred["num_heads"]),
        attn_drop=resolved_attn_drop,
        proj_drop=resolved_proj_drop,
        num_classes=resolved_num_classes,
        bottleneck_dim=int(inferred["bottleneck_dim"]),
        in_context_len=int(inferred["in_context_len"]),
        in_context_start=int(inferred.get("in_context_start", 0)),
        preset_name=None,
    )
    return model, checkpoint, inferred


def save_diffusers_jit_bundle(
    output_dir: str | Path,
    model: JiTModel,
    scheduler: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(transformer_dir, safe_serialization=True)
    if scheduler is not None:
        scheduler.save_pretrained(scheduler_dir)
    if metadata is not None:
        (output_dir / "bundle_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "transformer_dir": str(transformer_dir),
        "scheduler_dir": str(scheduler_dir),
        "report_path": str(output_dir / "bundle_report.json"),
    }