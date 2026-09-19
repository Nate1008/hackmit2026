"""Raw-video preprocessing for multi-object ShapeR reconstruction.

The pipeline deliberately stays local and synchronous for a hackathon demo:
sample video frames, track object instances with SAM3, recover metric geometry
and camera poses with Depth Anything 3, then emit one ShapeR PKL per object.
"""

from __future__ import annotations

import gc
import io
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fpsample
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


DEFAULT_DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_SAM3_MODEL = "facebook/sam3"


@dataclass
class ObjectTrack:
    object_id: int
    label: str
    masks: list[np.ndarray | None]
    scores: list[float] = field(default_factory=list)

    @property
    def visibility(self) -> int:
        return sum(mask is not None and np.any(mask) for mask in self.masks)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return value[:48] or "object"


def _encode_image(array: np.ndarray, fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    max_frames: int = 16,
    max_edge: int = 960,
) -> tuple[list[np.ndarray], list[Path], dict]:
    """Uniformly sample RGB frames while retaining deterministic frame indices."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if max_frames < 2:
        raise ValueError("At least two video frames are required")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not decode video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        capture.release()
        raise ValueError("Video reports no decodable frames")

    indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count))
    indices = sorted(set(int(round(index)) for index in indices))
    wanted = set(indices)
    frames: list[np.ndarray] = []
    sampled_indices: list[int] = []
    index = 0
    while index <= indices[-1]:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            if max(height, width) > max_edge:
                scale = max_edge / max(height, width)
                rgb = cv2.resize(
                    rgb,
                    (int(round(width * scale)), int(round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(rgb)
            sampled_indices.append(index)
        index += 1
    capture.release()
    if len(frames) < 2:
        raise ValueError("Could not extract at least two frames from the video")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ordinal, (frame_index, frame) in enumerate(zip(sampled_indices, frames)):
        path = output_dir / f"{ordinal:03d}_{frame_index:06d}.jpg"
        Image.fromarray(frame).save(path, quality=92)
        paths.append(path)

    metadata = {
        "source": str(video_path.resolve()),
        "source_frame_count": frame_count,
        "source_fps": fps,
        "sampled_indices": sampled_indices,
        "sampled_frame_count": len(frames),
    }
    return frames, paths, metadata


def _track_rank(track: ObjectTrack, frame_area: int) -> float:
    areas = [np.count_nonzero(mask) / frame_area for mask in track.masks if mask is not None]
    return track.visibility * (float(np.median(areas)) if areas else 0.0) * max(track.mean_score, 0.1)


def _mean_track_iou(left: ObjectTrack, right: ObjectTrack) -> float:
    values = []
    for mask_a, mask_b in zip(left.masks, right.masks):
        if mask_a is None or mask_b is None:
            continue
        intersection = np.count_nonzero(mask_a & mask_b)
        union = np.count_nonzero(mask_a | mask_b)
        if union:
            values.append(intersection / union)
    return float(np.mean(values)) if values else 0.0


def _split_disconnected_tracks(
    tracks: list[ObjectTrack],
    frame_area: int,
    minimum_visibility: int,
) -> list[ObjectTrack]:
    split_tracks: list[ObjectTrack] = []
    min_component_area = max(96, int(frame_area * 0.0015))
    for track in tracks:
        branches: list[dict] = []
        for frame_index, mask in enumerate(track.masks):
            if mask is None:
                continue
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            components = []
            height, width = mask.shape
            for component_id in range(1, count):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area < min_component_area:
                    continue
                component_mask = labels == component_id
                centroid = np.asarray(centroids[component_id], dtype=float) / np.array(
                    [width, height], dtype=float
                )
                components.append((component_mask, centroid, area))
            components.sort(key=lambda item: item[2], reverse=True)
            assigned_branches: set[int] = set()
            for component_mask, centroid, area in components:
                candidates = []
                for branch_index, branch in enumerate(branches):
                    if branch_index in assigned_branches:
                        continue
                    gap = frame_index - branch["last_frame"]
                    if gap > 3:
                        continue
                    distance = float(np.linalg.norm(centroid - branch["centroid"]))
                    area_change = abs(np.log(max(area, 1) / max(branch["area"], 1)))
                    candidates.append((distance + 0.08 * area_change, branch_index))
                if candidates and min(candidates)[0] < 0.42:
                    _, branch_index = min(candidates)
                    branch = branches[branch_index]
                else:
                    branch_index = len(branches)
                    branch = {
                        "masks": [None] * len(track.masks),
                        "centroid": centroid,
                        "area": area,
                        "last_frame": frame_index,
                    }
                    branches.append(branch)
                branch["masks"][frame_index] = component_mask
                branch["centroid"] = centroid
                branch["area"] = area
                branch["last_frame"] = frame_index
                assigned_branches.add(branch_index)

        valid = [
            branch
            for branch in branches
            if sum(mask is not None for mask in branch["masks"]) >= minimum_visibility
        ]
        if len(valid) < 2:
            split_tracks.append(track)
            continue
        valid.sort(
            key=lambda branch: np.median(
                [np.count_nonzero(mask) for mask in branch["masks"] if mask is not None]
            ),
            reverse=True,
        )
        for branch_index, branch in enumerate(valid):
            split_tracks.append(
                ObjectTrack(
                    object_id=track.object_id * 1000 + branch_index,
                    label=track.label,
                    masks=branch["masks"],
                    scores=list(track.scores),
                )
            )
    return split_tracks


def _deduplicate_tracks(
    tracks: list[ObjectTrack], frame_area: int, iou_threshold: float = 0.72
) -> list[ObjectTrack]:
    kept: list[ObjectTrack] = []
    for candidate in sorted(tracks, key=lambda x: _track_rank(x, frame_area), reverse=True):
        if all(
            _mean_track_iou(candidate, existing)
            < (0.92 if candidate.label == existing.label else iou_threshold)
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def segment_video_sam3(
    frames: list[np.ndarray],
    prompts: list[str],
    model_id: str = DEFAULT_SAM3_MODEL,
    device: str = "cuda",
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.80,
    max_objects: int = 8,
) -> list[ObjectTrack]:
    """Detect and track all instances matching one or more SAM3 concepts."""
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    if not prompts:
        prompts = ["object"]
    torch_device = torch.device(device)
    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    try:
        processor = Sam3VideoProcessor.from_pretrained(model_id)
        model = Sam3VideoModel.from_pretrained(model_id, dtype=dtype).to(torch_device).eval()
    except OSError as error:
        raise RuntimeError(
            "SAM3 weights are gated. Request access to facebook/sam3, then run "
            "`HF_HOME=.cache/huggingface hf auth login`."
        ) from error

    session = processor.init_video_session(
        video=frames,
        inference_device=torch_device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    session = processor.add_text_prompt(
        inference_session=session,
        text=prompts if len(prompts) > 1 else prompts[0],
    )

    height, width = frames[0].shape[:2]
    frame_area = height * width
    by_id: dict[int, ObjectTrack] = {}
    with torch.inference_mode():
        iterator = model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(frames) - 1,
        )
        for raw in iterator:
            frame_index = int(raw.frame_idx)
            result = processor.postprocess_outputs(session, raw)
            object_ids = _as_numpy(result["object_ids"]).astype(int).tolist()
            masks = _as_numpy(result["masks"]) > 0
            scores = _as_numpy(result.get("scores", np.ones(len(object_ids)))).reshape(-1)
            id_to_label = {
                int(object_id): label
                for label, ids in result.get("prompt_to_obj_ids", {}).items()
                for object_id in ids
            }
            for row, object_id in enumerate(object_ids):
                mask = np.squeeze(masks[row]).astype(bool)
                ratio = np.count_nonzero(mask) / frame_area
                if ratio < min_area_ratio or ratio > max_area_ratio:
                    continue
                label = id_to_label.get(int(object_id), "object")
                track = by_id.setdefault(
                    object_id,
                    ObjectTrack(object_id, label, [None] * len(frames)),
                )
                track.masks[frame_index] = mask
                track.scores.append(float(scores[row]) if row < len(scores) else 1.0)
    del model, processor, session
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    minimum_visibility = max(2, len(frames) // 8)
    tracks = [track for track in by_id.values() if track.visibility >= minimum_visibility]
    tracks = _split_disconnected_tracks(tracks, frame_area, minimum_visibility)
    tracks = _deduplicate_tracks(tracks, frame_area)
    tracks.sort(key=lambda x: _track_rank(x, frame_area), reverse=True)
    return tracks[:max_objects]


def load_tracks_npz(path: str | Path, frame_count: int) -> list[ObjectTrack]:
    """Load externally generated masks for debugging without SAM3.

    The NPZ must contain ``masks`` shaped [objects, frames, height, width] and
    may contain a string ``labels`` array.
    """
    archive = np.load(path, allow_pickle=False)
    masks = archive["masks"].astype(bool)
    if masks.ndim != 4 or masks.shape[1] != frame_count:
        raise ValueError("masks NPZ must have shape [objects, sampled_frames, H, W]")
    labels = archive["labels"].astype(str).tolist() if "labels" in archive else []
    return [
        ObjectTrack(
            object_id=index,
            label=labels[index] if index < len(labels) else f"object_{index:02d}",
            masks=[mask for mask in object_masks],
            scores=[1.0] * frame_count,
        )
        for index, object_masks in enumerate(masks)
    ]


def infer_da3_geometry(
    frame_paths: list[Path],
    model_id: str = DEFAULT_DA3_MODEL,
    device: str = "cuda",
):
    """Jointly infer metric depth, intrinsics and world-to-camera poses."""
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id).to(torch.device(device)).eval()
    with torch.inference_mode():
        prediction = model.inference(
            image=[str(path) for path in frame_paths],
            ref_view_strategy="middle",
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction


def _to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = _as_numpy(extrinsics).astype(np.float64)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    output = np.repeat(np.eye(4, dtype=np.float64)[None], len(extrinsics), axis=0)
    output[:, :3, :4] = extrinsics
    return output


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 0.999999:
        return np.eye(3)
    if cosine < -0.999999:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis -= source * np.dot(axis, source)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(resized, kernel, iterations=1).astype(bool)


def _masked_world_points(
    depth: np.ndarray,
    confidence: np.ndarray | None,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0)
    if confidence is not None and np.any(valid):
        threshold = np.percentile(confidence[valid], 15)
        valid &= confidence >= threshold
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=0)
    rays = np.linalg.inv(intrinsic) @ pixels
    camera_points = rays * depth[ys, xs][None]
    camera_h = np.vstack([camera_points, np.ones((1, camera_points.shape[1]))])
    return (np.linalg.inv(world_to_camera) @ camera_h)[:3].T.astype(np.float32)


def _robust_cloud_summary(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a stable world-space center and radius for one mask observation."""
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 16:
        raise ValueError("Too few valid points for a mask observation")
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    cutoff = np.percentile(distances, 90.0)
    inliers = points[distances <= cutoff]
    if len(inliers) >= 16:
        center = np.median(inliers, axis=0)
    low, high = np.percentile(inliers, [10.0, 90.0], axis=0)
    radius = max(float(np.linalg.norm(high - low) * 0.5), 0.03)
    return center.astype(np.float64), radius


def _component_masks(mask: np.ndarray, minimum_area: int) -> list[np.ndarray]:
    """Split disconnected regions and cut thin bridges between nearby instances.

    SAM occasionally joins adjacent objects with a narrow mask bridge. A single
    connected-component pass cannot separate those objects, so use conservative
    multi-scale erosion to find stable seeds and assign the original foreground
    pixels back to their nearest seed.
    """
    binary = mask.astype(np.uint8)
    original_area = int(np.count_nonzero(binary))
    if original_area < minimum_area:
        return []

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = [
        labels == component_id
        for component_id in range(1, count)
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if len(components) != 1:
        return components

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    minimum_seed_area = max(12, minimum_area // 2)
    for iterations in (1, 2, 3):
        eroded = cv2.erode(binary, kernel, iterations=iterations)
        seed_count, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
            eroded, connectivity=8
        )
        seed_ids = [
            component_id
            for component_id in range(1, seed_count)
            if int(seed_stats[component_id, cv2.CC_STAT_AREA]) >= minimum_seed_area
        ]
        if len(seed_ids) < 2:
            continue
        seed_ids.sort(
            key=lambda component_id: int(seed_stats[component_id, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        seed_ids = seed_ids[:6]
        distances = np.stack(
            [
                cv2.distanceTransform(
                    (seed_labels != component_id).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                )
                for component_id in seed_ids
            ]
        )
        nearest = np.argmin(distances, axis=0)
        split = [
            mask & (nearest == seed_index)
            for seed_index in range(len(seed_ids))
        ]
        split = [part for part in split if np.count_nonzero(part) >= minimum_area]
        if len(split) >= 2:
            second_largest = sorted(
                (int(np.count_nonzero(part)) for part in split), reverse=True
            )[1]
            if second_largest >= max(minimum_area, int(original_area * 0.05)):
                return split

    return components


def refine_tracks_by_world_geometry(
    tracks: list[ObjectTrack],
    prediction,
    *,
    max_objects: int | None = None,
) -> list[ObjectTrack]:
    """Split SAM tracks using DA3 world geometry and discard identity-switch outliers.

    SAM can reuse one semantic track for two visually similar instances. 2D centroid
    matching cannot reliably catch that under camera motion, so observations are
    associated again using their reconstructed, static world positions.
    """
    depths = _as_numpy(prediction.depth)
    confidences = (
        _as_numpy(prediction.conf)
        if getattr(prediction, "conf", None) is not None
        else None
    )
    intrinsics = _as_numpy(prediction.intrinsics).astype(np.float64)
    world_to_cameras = _to_homogeneous(prediction.extrinsics)
    depth_height, depth_width = depths.shape[-2:]
    minimum_visibility = max(2, len(depths) // 8)
    minimum_area = max(16, int(depth_height * depth_width * 0.0006))
    refined: list[ObjectTrack] = []

    for track in tracks:
        observations_by_frame: dict[int, list[dict]] = {}
        for frame_index, raw_mask in enumerate(track.masks):
            if raw_mask is None:
                continue
            resized = _resize_mask(raw_mask, depth_width, depth_height)
            for component in _component_masks(resized, minimum_area):
                points = _masked_world_points(
                    depths[frame_index],
                    confidences[frame_index] if confidences is not None else None,
                    component,
                    intrinsics[frame_index],
                    world_to_cameras[frame_index],
                )
                try:
                    center, radius = _robust_cloud_summary(points)
                except ValueError:
                    continue
                observations_by_frame.setdefault(frame_index, []).append(
                    {"mask": component, "center": center, "radius": radius}
                )

        branches: list[dict] = []
        for frame_index in sorted(observations_by_frame):
            observations = observations_by_frame[frame_index]
            if not branches:
                for observation in observations:
                    branches.append(
                        {
                            "masks": [None] * len(track.masks),
                            "centers": [observation["center"]],
                            "radii": [observation["radius"]],
                        }
                    )
                    branches[-1]["masks"][frame_index] = observation["mask"]
                continue

            branch_centers = np.asarray(
                [np.median(branch["centers"], axis=0) for branch in branches]
            )
            branch_radii = np.asarray(
                [np.median(branch["radii"]) for branch in branches]
            )
            observation_centers = np.asarray([item["center"] for item in observations])
            distances = np.linalg.norm(
                branch_centers[:, None, :] - observation_centers[None, :, :], axis=2
            )
            rows, columns = linear_sum_assignment(distances)
            assigned_observations: set[int] = set()
            for branch_index, observation_index in zip(rows.tolist(), columns.tolist()):
                observation = observations[observation_index]
                gate = max(
                    0.18,
                    min(
                        0.85,
                        0.72 * (branch_radii[branch_index] + observation["radius"]),
                    ),
                )
                if distances[branch_index, observation_index] > gate:
                    continue
                branch = branches[branch_index]
                branch["masks"][frame_index] = observation["mask"]
                branch["centers"].append(observation["center"])
                branch["radii"].append(observation["radius"])
                assigned_observations.add(observation_index)

            for observation_index, observation in enumerate(observations):
                if observation_index in assigned_observations:
                    continue
                branch = {
                    "masks": [None] * len(track.masks),
                    "centers": [observation["center"]],
                    "radii": [observation["radius"]],
                }
                branch["masks"][frame_index] = observation["mask"]
                branches.append(branch)

        valid_branches = [
            branch
            for branch in branches
            if sum(mask is not None for mask in branch["masks"]) >= minimum_visibility
        ]
        valid_branches.sort(
            key=lambda branch: sum(
                np.count_nonzero(mask)
                for mask in branch["masks"]
                if mask is not None
            ),
            reverse=True,
        )
        for branch_index, branch in enumerate(valid_branches):
            refined.append(
                ObjectTrack(
                    object_id=track.object_id * 1000 + branch_index,
                    label=track.label,
                    masks=branch["masks"],
                    scores=list(track.scores),
                )
            )

    if not refined:
        return []
    frame_area = int(depth_height * depth_width)
    refined = _deduplicate_tracks(refined, frame_area)
    refined.sort(key=lambda item: _track_rank(item, frame_area), reverse=True)
    return refined[:max_objects] if max_objects is not None else refined


def _filter_and_sample(points: np.ndarray, max_points: int = 2048) -> np.ndarray:
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 16:
        raise ValueError("Too few valid object points after depth fusion")
    if len(points) > 100_000:
        rng = np.random.default_rng(42)
        points = points[rng.choice(len(points), 100_000, replace=False)]
    if len(points) >= 64:
        tree = cKDTree(points)
        neighbor_distances, _ = tree.query(points, k=min(9, len(points)))
        local_scale = neighbor_distances[:, -1]
        median_scale = float(np.median(local_scale))
        mad_scale = float(np.median(np.abs(local_scale - median_scale)))
        density_limit = median_scale + max(4.5 * mad_scale, median_scale * 1.5)
        dense = points[local_scale <= density_limit]
        if len(dense) >= 32:
            points = dense
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    points = points[distances <= np.percentile(distances, 99.0)]
    if len(points) >= 64:
        low, high = np.percentile(points, [0.5, 99.5], axis=0)
        trimmed = points[np.all((points >= low) & (points <= high), axis=1)]
        if len(trimmed) >= 32:
            points = trimmed
    if len(points) > max_points:
        indices = fpsample.fps_sampling(points.astype(np.float32), max_points)
        points = points[indices]
    return points.astype(np.float32)


def build_object_pkls(
    frames: list[np.ndarray],
    tracks: list[ObjectTrack],
    prediction,
    output_dir: str | Path,
) -> tuple[list[Path], list[dict]]:
    """Fuse each SAM3 track into its own metric, world-placeable ShapeR sample."""
    depths = _as_numpy(prediction.depth)
    confidences = _as_numpy(prediction.conf) if getattr(prediction, "conf", None) is not None else None
    intrinsics = _as_numpy(prediction.intrinsics).astype(np.float64)
    world_to_cameras = _to_homogeneous(prediction.extrinsics)
    if len(depths) != len(frames):
        raise ValueError("DA3 returned a different number of views than supplied")
    camera_to_worlds = np.linalg.inv(world_to_cameras)

    average_up = np.mean(-camera_to_worlds[:, :3, 1], axis=0)
    if np.linalg.norm(average_up) < 1e-6:
        average_up = np.array([0.0, -1.0, 0.0])
    up_rotation = _rotation_between(average_up, np.array([0.0, 0.0, 1.0]))
    transform_up = np.eye(4, dtype=np.float64)
    transform_up[:3, :3] = up_rotation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    manifest_objects: list[dict] = []
    depth_height, depth_width = depths.shape[-2:]

    for ordinal, track in enumerate(tracks):
        per_view_masks: list[np.ndarray] = []
        point_chunks = []
        valid_view_indices = []
        for frame_index, raw_mask in enumerate(track.masks):
            if raw_mask is None:
                continue
            mask = _resize_mask(raw_mask, depth_width, depth_height)
            if np.count_nonzero(mask) < 16:
                continue
            points = _masked_world_points(
                depths[frame_index],
                confidences[frame_index] if confidences is not None else None,
                mask,
                intrinsics[frame_index],
                world_to_cameras[frame_index],
            )
            if len(points) < 16:
                continue
            valid_view_indices.append(frame_index)
            per_view_masks.append(mask)
            point_chunks.append(points)
        if len(valid_view_indices) < 2:
            continue

        try:
            world_points = _filter_and_sample(np.concatenate(point_chunks, axis=0))
        except ValueError:
            continue
        points_up = (up_rotation @ world_points.T).T
        minimum = points_up.min(axis=0)
        maximum = points_up.max(axis=0)
        center = (minimum + maximum) / 2
        centered_points = points_up - center
        bounds = np.maximum((maximum - minimum) / 2, 1e-3)
        transform_center = np.eye(4, dtype=np.float64)
        transform_center[:3, 3] = -center
        model_from_world = transform_center @ transform_up

        image_data = []
        mask_data = []
        sample_camera_to_worlds = []
        sample_intrinsics = []
        for local_index, frame_index in enumerate(valid_view_indices):
            resized = cv2.resize(
                frames[frame_index], (depth_width, depth_height), interpolation=cv2.INTER_AREA
            )
            grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            image_data.append(_encode_image(grayscale, "JPEG"))
            mask_data.append(_encode_image(per_view_masks[local_index].astype(np.uint8) * 255, "PNG"))
            sample_camera_to_worlds.append(
                (model_from_world @ camera_to_worlds[frame_index]).astype(np.float32)
            )
            sample_intrinsics.append(intrinsics[frame_index].astype(np.float32))

        label = _safe_name(track.label)
        name = f"object_{ordinal:02d}_{label}"
        sample = {
            "points_model": torch.from_numpy(centered_points.astype(np.float32)),
            "bounds": torch.from_numpy(bounds.astype(np.float32)),
            "T_model_world": torch.from_numpy(model_from_world.astype(np.float32)),
            "inv_dist_std": torch.zeros(len(centered_points), dtype=torch.float32),
            "dist_std": torch.zeros(len(centered_points), dtype=torch.float32),
            "image_data": image_data,
            "camera_to_worlds": [torch.from_numpy(value) for value in sample_camera_to_worlds],
            "camera_params": [torch.from_numpy(value) for value in sample_intrinsics],
            "mask_data": mask_data,
            "frame_indices": valid_view_indices,
            "caption": track.label,
            "experimental_dav3": True,
            "source_object_id": track.object_id,
        }
        path = output_dir / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(sample, handle)
        paths.append(path)
        manifest_objects.append(
            {
                "name": name,
                "label": track.label,
                "sam3_object_id": track.object_id,
                "visible_views": len(valid_view_indices),
                "point_count": len(centered_points),
                "bounds_m": bounds.tolist(),
                "pkl": str(path),
            }
        )
    return paths, manifest_objects


def preprocess_video(
    video_path: str | Path,
    work_dir: str | Path,
    prompts: list[str] | None = None,
    max_frames: int = 16,
    max_objects: int = 8,
    masks_npz: str | Path | None = None,
    sam3_model: str = DEFAULT_SAM3_MODEL,
    da3_model: str = DEFAULT_DA3_MODEL,
) -> tuple[list[Path], Path]:
    work_dir = Path(work_dir)
    frames, frame_paths, metadata = extract_video_frames(
        video_path, work_dir / "frames", max_frames=max_frames
    )
    if masks_npz:
        tracks = load_tracks_npz(masks_npz, len(frames))
    else:
        tracks = segment_video_sam3(
            frames,
            prompts or ["object"],
            model_id=sam3_model,
            max_objects=max_objects,
        )
    if not tracks:
        raise RuntimeError("SAM3 did not find any persistent object tracks")

    prediction = infer_da3_geometry(frame_paths, model_id=da3_model)
    tracks = refine_tracks_by_world_geometry(
        tracks, prediction, max_objects=max_objects
    )
    if not tracks:
        raise RuntimeError("No spatially consistent object tracks survived 3D refinement")
    confidence = getattr(prediction, "conf", None)
    geometry_path = work_dir / "geometry.npz"
    np.savez_compressed(
        geometry_path,
        depths=_as_numpy(prediction.depth).astype(np.float32),
        confidences=(
            _as_numpy(confidence).astype(np.float32)
            if confidence is not None
            else np.empty(0, dtype=np.float32)
        ),
        intrinsics=_as_numpy(prediction.intrinsics).astype(np.float32),
        extrinsics=_as_numpy(prediction.extrinsics).astype(np.float32),
    )
    pkl_paths, objects = build_object_pkls(
        frames, tracks, prediction, work_dir / "objects"
    )
    if not pkl_paths:
        raise RuntimeError("No object had enough consistent depth points for ShapeR")

    manifest = {
        **metadata,
        "sam3_model": sam3_model,
        "da3_model": da3_model,
        "prompts": prompts or ["object"],
        "geometry": str(geometry_path),
        "objects": objects,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pkl_paths, manifest_path

