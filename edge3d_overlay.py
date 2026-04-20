import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from edge3d.generation.pipeline import DEFAULT_EDGE3D_ALIGNMENT, export_uid_overlay, load_alignment_from_report


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be R,G,B")
    try:
        color = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must contain integers") from exc
    if any(channel < 0 or channel > 255 for channel in color):
        raise argparse.ArgumentTypeError("Each color channel must be in [0, 255]")
    return color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an Edge3D/Objectaverse overlay for a single UID.")
    parser.add_argument("uid", help="Object UID shared by models/<uid>.npz and Objaverse")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Suffix controls export format, for example .obj or .glb.",
    )
    parser.add_argument(
        "--alignment-report",
        default=None,
        help="Optional alignment_report.json. If omitted, use the verified default global alignment.",
    )
    parser.add_argument("--edge-radius", type=float, default=0.01)
    parser.add_argument("--edge-sections", type=int, default=6)
    parser.add_argument("--download-processes", type=int, default=4)
    parser.add_argument("--mesh-color", type=parse_rgb, default=(160, 160, 168), help="Mesh RGB, default 160,160,168")
    parser.add_argument("--edge-color", type=parse_rgb, default=(0, 190, 255), help="Edge RGB, default 0,190,255")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else Path("demo_outputs") / "edge3d_uid_overlays" / args.uid / f"{args.uid}_overlay.obj"
    alignment = load_alignment_from_report(args.alignment_report) if args.alignment_report else DEFAULT_EDGE3D_ALIGNMENT
    summary = export_uid_overlay(
        uid=args.uid,
        dataset_root=args.dataset_root,
        output_path=output_path,
        alignment=alignment,
        edge_radius=args.edge_radius,
        edge_sections=args.edge_sections,
        mesh_color=args.mesh_color,
        edge_color=args.edge_color,
        download_processes=args.download_processes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()