from types import SimpleNamespace

import numpy as np

from video_pipeline import ObjectTrack, refine_tracks_by_world_geometry


def _prediction(depths: np.ndarray) -> SimpleNamespace:
    frame_count, height, width = depths.shape
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
    intrinsics[:, 0, 0] = 10.0
    intrinsics[:, 1, 1] = 10.0
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    return SimpleNamespace(
        depth=depths.astype(np.float32),
        conf=np.ones_like(depths, dtype=np.float32),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


def _square(height: int, width: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[4:12, x0:x1] = True
    return mask


def test_splits_identity_switches_that_are_far_apart_in_world_space():
    height = width = 16
    depths = np.stack(
        [np.full((height, width), depth) for depth in (2.0, 2.0, 6.0, 6.0)]
    )
    mask = _square(height, width, 4, 12)
    track = ObjectTrack(7, "box", [mask.copy() for _ in range(4)], [1.0] * 4)

    refined = refine_tracks_by_world_geometry([track], _prediction(depths))

    assert len(refined) == 2
    assert sorted(item.visibility for item in refined) == [2, 2]


def test_preserves_two_instances_present_in_the_same_frames():
    height = width = 24
    depths = np.full((4, height, width), 3.0, dtype=np.float32)
    left = _square(height, width, 2, 8)
    right = _square(height, width, 16, 22)
    combined = left | right
    track = ObjectTrack(3, "trash can", [combined.copy() for _ in range(4)], [1.0] * 4)

    refined = refine_tracks_by_world_geometry([track], _prediction(depths))

    assert len(refined) == 2
    assert [item.visibility for item in refined] == [4, 4]


def test_splits_nearby_instances_joined_by_a_thin_mask_bridge():
    height = width = 28
    depths = np.full((4, height, width), 3.0, dtype=np.float32)
    merged = np.zeros((height, width), dtype=bool)
    merged[4:20, 2:11] = True
    merged[4:20, 17:26] = True
    merged[10:14, 11:17] = True
    track = ObjectTrack(5, "box", [merged.copy() for _ in range(4)], [1.0] * 4)

    refined = refine_tracks_by_world_geometry([track], _prediction(depths))

    assert len(refined) == 2
    assert [item.visibility for item in refined] == [4, 4]
