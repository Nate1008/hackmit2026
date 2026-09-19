"""Turn one raw video into separately reconstructed ShapeR object meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh

from scene_postprocessing import build_room_shell, colorize_object_meshes
from shaper_runtime import ShapeRRuntime
from video_pipeline import preprocess_video


def _parse_prompts(values: list[str] | None) -> list[str]:
    prompts = []
    for value in values or ["object"]:
        prompts.extend(item.strip() for item in value.split(",") if item.strip())
    return prompts or ["object"]


def _combine_meshes(mesh_paths: list[Path], output_path: Path) -> None:
    """Combine colored meshes without flattening their named scene nodes."""
    combined = trimesh.Scene()
    for path in mesh_paths:
        source = trimesh.load(path, force="scene")
        nodes = list(source.graph.nodes_geometry)
        for node in nodes:
            transform, geometry_name = source.graph[node]
            mesh = source.geometry[geometry_name].copy()
            name = path.stem if len(nodes) == 1 else str(node)
            combined.add_geometry(mesh, node_name=name, geom_name=name, transform=transform)
    combined.export(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/video_reconstruction"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--preset", choices=("speed", "balance", "quality"), default="balance")
    parser.add_argument("--masks-npz", type=Path)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    work_dir = (args.work_dir or output_dir / "work").resolve()
    samples, manifest_path = preprocess_video(
        args.video,
        work_dir,
        prompts=_parse_prompts(args.prompt),
        max_frames=args.max_frames,
        max_objects=args.max_objects,
        masks_npz=args.masks_npz,
    )
    if args.preprocess_only:
        print(json.dumps({"manifest": str(manifest_path), "samples": [str(x) for x in samples]}))
        return
    meshes = ShapeRRuntime().reconstruct_many(samples, output_dir / "objects", args.preset)
    color_statistics = colorize_object_meshes(samples, meshes, work_dir)
    room_path, room_nodes = build_room_shell(
        samples, work_dir, output_dir / "room_shell.glb"
    )
    scene_path = output_dir / "scene.glb"
    _combine_meshes([*meshes, room_path], scene_path)
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    result["meshes"] = [str(path.resolve()) for path in meshes]
    result["scene"] = str(scene_path.resolve())
    result["room_shell"] = str(room_path.resolve())
    result["room_nodes"] = room_nodes
    result["object_colors"] = color_statistics
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "scene": result["scene"],
                "meshes": result["meshes"],
            }
        )
    )


if __name__ == "__main__":
    main()

