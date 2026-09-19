"""Color projection, room fusion and glTF coordinate conversion."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from video_pipeline import _rotation_between, _to_homogeneous


Z_UP_TO_GLTF_Y_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _decode_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("L")) > 0


def _load_bundle(work_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(work_dir) / "geometry.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DA3 geometry bundle: {path}")
    archive = np.load(path)
    return {key: archive[key] for key in archive.files}


def _world_to_gltf(extrinsics: np.ndarray) -> np.ndarray:
    world_to_cameras = _to_homogeneous(extrinsics)
    camera_to_worlds = np.linalg.inv(world_to_cameras)
    average_up = np.mean(-camera_to_worlds[:, :3, 1], axis=0)
    if np.linalg.norm(average_up) < 1e-6:
        average_up = np.array([0.0, -1.0, 0.0])
    world_to_z_up = np.eye(4, dtype=np.float64)
    world_to_z_up[:3, :3] = _rotation_between(
        average_up, np.array([0.0, 0.0, 1.0])
    )
    return Z_UP_TO_GLTF_Y_UP @ world_to_z_up


def _load_rgb_frames(work_dir: str | Path, width: int, height: int) -> list[np.ndarray]:
    paths = sorted((Path(work_dir) / "frames").glob("*.jpg"))
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode extracted frame: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA))
    return frames


def _project_object_colors(
    mesh: trimesh.Trimesh,
    sample: dict,
    frames: list[np.ndarray],
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[np.ndarray, float]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    best_score = np.full(len(vertices), np.inf, dtype=np.float64)
    colors = np.zeros((len(vertices), 3), dtype=np.uint8)
    fallback_colors = []

    frame_indices = sample.get("frame_indices")
    if frame_indices is None:
        raise ValueError("Object sample predates multi-view color metadata")
    for local_index, frame_index in enumerate(frame_indices):
        frame_index = int(frame_index)
        mask = _decode_mask(sample["mask_data"][local_index])
        frame = frames[frame_index]
        depth = depths[frame_index]
        height, width = depth.shape
        if mask.shape != depth.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if np.any(mask):
            fallback_colors.append(frame[mask])

        world_to_camera = _to_homogeneous(extrinsics[[frame_index]])[0]
        vertices_h = np.column_stack([vertices, np.ones(len(vertices))])
        camera_points = (world_to_camera @ vertices_h.T).T[:, :3]
        z = camera_points[:, 2]
        projected = (intrinsics[frame_index] @ camera_points.T).T
        u = np.rint(projected[:, 0] / np.maximum(projected[:, 2], 1e-8)).astype(int)
        v = np.rint(projected[:, 1] / np.maximum(projected[:, 2], 1e-8)).astype(int)
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            continue
        indices = indices[mask[v[indices], u[indices]]]
        if len(indices) == 0:
            continue

        observed_depth = depth[v[indices], u[indices]]
        residual = np.abs(z[indices] - observed_depth) / np.maximum(observed_depth, 0.1)
        indices = indices[np.isfinite(residual) & (residual < 0.35)]
        if len(indices) == 0:
            continue
        residual = np.abs(z[indices] - depth[v[indices], u[indices]]) / np.maximum(
            depth[v[indices], u[indices]], 0.1
        )

        camera_center = np.linalg.inv(world_to_camera)[:3, 3]
        view_vectors = camera_center[None] - vertices[indices]
        view_vectors /= np.maximum(np.linalg.norm(view_vectors, axis=1, keepdims=True), 1e-8)
        facing = np.abs(np.sum(normals[indices] * view_vectors, axis=1))
        score = residual + 0.12 * (1.0 - facing)
        update = score < best_score[indices]
        chosen = indices[update]
        best_score[chosen] = score[update]
        colors[chosen] = frame[v[chosen], u[chosen]]

    colored = np.isfinite(best_score)
    direct_fraction = float(np.mean(colored))
    if np.any(colored) and np.any(~colored):
        tree = cKDTree(vertices[colored])
        _, nearest = tree.query(vertices[~colored], k=1)
        colors[~colored] = colors[colored][nearest]
    elif not np.any(colored):
        if fallback_colors:
            fallback = np.median(np.concatenate(fallback_colors, axis=0), axis=0)
        else:
            fallback = np.array([160, 160, 160])
        colors[:] = np.asarray(fallback, dtype=np.uint8)
    return colors, direct_fraction


def colorize_object_meshes(
    sample_paths: list[str | Path],
    mesh_paths: list[str | Path],
    work_dir: str | Path,
) -> list[dict]:
    bundle = _load_bundle(work_dir)
    depths = bundle["depths"]
    intrinsics = bundle["intrinsics"]
    extrinsics = bundle["extrinsics"]
    height, width = depths.shape[-2:]
    frames = _load_rgb_frames(work_dir, width, height)
    world_to_gltf = _world_to_gltf(extrinsics)
    statistics = []

    for sample_path, mesh_path in zip(sample_paths, mesh_paths):
        with Path(sample_path).open("rb") as handle:
            sample = pickle.load(handle)
        mesh = trimesh.load_mesh(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        colors, direct_fraction = _project_object_colors(
            mesh, sample, frames, depths, intrinsics, extrinsics
        )
        mesh.visual.vertex_colors = np.column_stack(
            [colors, np.full(len(colors), 255, dtype=np.uint8)]
        )
        mesh.apply_transform(world_to_gltf)
        mesh.export(mesh_path, include_normals=True)
        statistics.append(
            {
                "name": Path(mesh_path).stem,
                "direct_color_fraction": direct_fraction,
            }
        )
    return statistics


def _foreground_masks(
    sample_paths: list[str | Path],
    frame_count: int,
    width: int,
    height: int,
) -> list[np.ndarray]:
    unions = [np.zeros((height, width), dtype=np.uint8) for _ in range(frame_count)]
    for sample_path in sample_paths:
        with Path(sample_path).open("rb") as handle:
            sample = pickle.load(handle)
        for local_index, frame_index in enumerate(sample.get("frame_indices", [])):
            mask = _decode_mask(sample["mask_data"][local_index])
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            unions[int(frame_index)] |= mask.astype(np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    return [cv2.dilate(mask, kernel, iterations=1).astype(bool) for mask in unions]


def _clean_room_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[labels] < 150)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def build_room_shell(
    sample_paths: list[str | Path],
    work_dir: str | Path,
    output_path: str | Path,
    voxel_length: float = 0.035,
) -> tuple[Path, list[str]]:
    bundle = _load_bundle(work_dir)
    depths = bundle["depths"].astype(np.float32)
    confidences = bundle.get("confidences")
    intrinsics = bundle["intrinsics"]
    extrinsics = bundle["extrinsics"]
    frame_count, height, width = depths.shape
    frames = _load_rgb_frames(work_dir, width, height)
    foreground = _foreground_masks(sample_paths, frame_count, width, height)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=voxel_length * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    depth_trunc = float(np.nanpercentile(depths[np.isfinite(depths)], 99.5) + 0.5)
    world_to_cameras = _to_homogeneous(extrinsics)
    for index in range(frame_count):
        depth = depths[index].copy()
        valid = np.isfinite(depth) & (depth > 0) & ~foreground[index]
        if confidences is not None and confidences.size:
            confidence = confidences[index]
            threshold = np.percentile(confidence[valid], 10) if np.any(valid) else np.inf
            valid &= confidence >= threshold
        depth[~valid] = 0
        color_image = o3d.geometry.Image(np.ascontiguousarray(frames[index]))
        depth_image = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics[index, 0, 0]),
            float(intrinsics[index, 1, 1]),
            float(intrinsics[index, 0, 2]),
            float(intrinsics[index, 1, 2]),
        )
        volume.integrate(rgbd, intrinsic, world_to_cameras[index])

    room_o3d = _clean_room_mesh(volume.extract_triangle_mesh())
    if len(room_o3d.triangles) == 0:
        raise RuntimeError("TSDF did not produce a room mesh")
    room = trimesh.Trimesh(
        vertices=np.asarray(room_o3d.vertices),
        faces=np.asarray(room_o3d.triangles),
        vertex_colors=np.asarray(room_o3d.vertex_colors),
        process=False,
    )
    room.apply_transform(_world_to_gltf(extrinsics))
    centers = room.triangles_center
    normals = room.face_normals
    camera_centers = np.linalg.inv(world_to_cameras)[:, :3, 3]
    camera_centers_h = np.column_stack([camera_centers, np.ones(len(camera_centers))])
    camera_y = (
        _world_to_gltf(extrinsics) @ camera_centers_h.T
    ).T[:, 1]
    floor_limit = float(np.median(camera_y) - 0.35)
    horizontal = np.abs(normals[:, 1]) > 0.68
    floor_faces = np.flatnonzero(horizontal & (centers[:, 1] < floor_limit))
    wall_faces = np.flatnonzero(np.abs(normals[:, 1]) < 0.55)
    used = np.zeros(len(room.faces), dtype=bool)
    used[floor_faces] = True
    used[wall_faces] = True
    other_faces = np.flatnonzero(~used)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    names = []
    for name, face_indices in (
        ("room_floor", floor_faces),
        ("room_walls", wall_faces),
        ("room_other", other_faces),
    ):
        if len(face_indices) == 0:
            continue
        part = room.submesh([face_indices], append=True, repair=False)
        if len(part.faces) == 0:
            continue
        scene.add_geometry(part, node_name=name, geom_name=name)
        names.append(name)
    scene.export(output_path)
    return output_path, names
