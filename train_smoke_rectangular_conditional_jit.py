import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models import RectangularConditionalJiTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal training smoke test for the rectangular conditional JiT model.")
    parser.add_argument(
        "--interaction-mode",
        default="all",
        choices=["sparse_xattn", "full_joint_mmdit", "all"],
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, nargs=2, default=(64, 128), metavar=("H", "W"))
    parser.add_argument("--condition-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--cond-bottleneck-dim", type=int, default=12)
    parser.add_argument("--cond-tower-depth", type=int, default=2)
    parser.add_argument("--cond-base-channels", type=int, default=8)
    parser.add_argument("--image-in-channels", type=int, default=1)
    parser.add_argument("--image-out-channels", type=int, default=2)
    parser.add_argument("--condition-channels-per-type", default="5,6,7")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recompute-global-after-joint", action="store_true")
    return parser.parse_args()


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_condition_channels(value: str) -> tuple[int, ...]:
    parts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 3:
        raise ValueError("condition_channels_per_type must contain exactly 3 comma-separated integers")
    return parts


def _build_target(sample: torch.Tensor, condition: torch.Tensor, out_channels: int) -> torch.Tensor:
    condition_resized = condition
    if condition.shape[-2:] != sample.shape[-2:]:
        condition_resized = F.interpolate(condition, size=sample.shape[-2:], mode="bilinear", align_corners=False)

    target_channels = []
    for channel_index in range(out_channels):
        if channel_index % 2 == 0:
            source = sample[:, channel_index % sample.shape[1]]
        else:
            source = condition_resized[:, channel_index % condition_resized.shape[1]]
        target_channels.append(source)
    return torch.stack(target_channels, dim=1).tanh()


def _make_batch(
    batch_size: int,
    image_in_channels: int,
    image_out_channels: int,
    input_size: tuple[int, int],
    condition_size: tuple[int, int],
    condition_channels_per_type: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample = torch.randn(batch_size, image_in_channels, input_size[0], input_size[1], device=device)
    condition = torch.randn(
        batch_size,
        max(condition_channels_per_type),
        condition_size[0],
        condition_size[1],
        device=device,
    )
    condition_type_ids = torch.arange(batch_size, device=device, dtype=torch.long) % len(condition_channels_per_type)
    timestep = torch.linspace(0.15, 0.85, batch_size, device=device)
    target = _build_target(sample, condition, image_out_channels)
    return sample, condition, condition_type_ids, timestep, target


def run_smoke(mode: str, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    input_size = tuple(int(value) for value in args.input_size)
    condition_size = tuple(int(value) for value in (args.condition_size or args.input_size))
    condition_channels_per_type = _parse_condition_channels(args.condition_channels_per_type)
    model = RectangularConditionalJiTModel(
        input_size=input_size,
        patch_size=args.patch_size,
        image_in_channels=args.image_in_channels,
        image_out_channels=args.image_out_channels,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        bottleneck_dim=args.bottleneck_dim,
        condition_size=condition_size,
        condition_channels_per_type=condition_channels_per_type,
        cond_base_channels=args.cond_base_channels,
        cond_bottleneck_dim=args.cond_bottleneck_dim,
        cond_tower_depth=args.cond_tower_depth,
        interaction_mode=mode,
        recompute_global_after_joint=args.recompute_global_after_joint,
        preset_name=None,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sample, condition, condition_type_ids, timestep, target = _make_batch(
        args.batch_size,
        args.image_in_channels,
        args.image_out_channels,
        input_size,
        condition_size,
        condition_channels_per_type,
        device,
    )

    losses: list[float] = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            sample=sample,
            timestep=timestep,
            condition=condition,
            condition_type_ids=condition_type_ids,
        )
        loss = F.mse_loss(output.sample, target)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)
        print(f"[{mode}] step {step + 1}/{args.steps} loss={loss_value:.6f}")

    loss_decreased = losses[-1] < losses[0]
    print(f"[{mode}] loss_decreased={loss_decreased} initial={losses[0]:.6f} final={losses[-1]:.6f}")
    return {
        "interaction_mode": mode,
        "losses": losses,
        "loss_decreased": loss_decreased,
        "input_size": list(input_size),
        "condition_size": list(condition_size),
        "patch_size": args.patch_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "device": str(device),
        "recompute_global_after_joint": bool(args.recompute_global_after_joint),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(_resolve_device(args.device))

    modes = [args.interaction_mode] if args.interaction_mode != "all" else ["sparse_xattn", "full_joint_mmdit"]
    results = [run_smoke(mode, args, device) for mode in modes]
    if not all(result["loss_decreased"] for result in results):
        raise RuntimeError("At least one smoke run did not reduce the loss")

    summary = {"seed": args.seed, "results": results}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()