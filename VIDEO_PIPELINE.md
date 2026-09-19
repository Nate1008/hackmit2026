# Raw video to separate object meshes

The upstream ShapeR release starts from an already prepared per-object PKL.
This project now supplies the missing raw-video preprocessing path.

## Run

```bash
eval "$(/home/ubuntu/ShapeR/.conda/bin/conda shell.bash hook)"
conda activate /home/ubuntu/ShapeR/.conda/envs/shaper
python video_to_3d.py input.mp4 \
  --output-dir output/my_video \
  --prompt chair,table,lamp \
  --max-frames 16 \
  --max-objects 8 \
  --preset balance
```

Result:

- `output/my_video/objects/*.glb`: one colored, Y-up mesh per tracked object
- `output/my_video/room_shell.glb`: colored floor, walls and remaining background
- `output/my_video/scene.glb`: all objects and room parts as separate named nodes
- `output/my_video/work/manifest.json`: labels, bounds, source frames and paths

Use prompts that describe likely scene contents for the best coverage.
`--prompt object` is the prompt-free fallback, but may split an object into parts.

## What runs

1. OpenCV uniformly samples and resizes video frames.
2. SAM3 detects and tracks all instances matching the supplied concepts.
3. Depth Anything 3 jointly estimates depth, camera intrinsics and camera poses.
4. Per-instance depth is fused in world coordinates and ShapeR generates each GLB.
5. Original RGB is projected onto object vertices with mask/depth occlusion tests.
6. Background RGB-D is TSDF-fused into floor, wall and other room meshes.
7. DA3 coordinates are converted through ShapeR Z-up into glTF-standard Y-up.

## Notes

- SAM3 access is gated; Hugging Face authentication is already project-local.
- Defaults retain 16 frames and at most 8 persistent objects.
- The first ShapeR request includes model loading and `torch.compile` warm-up.
- ShapeR and the selected DA3 checkpoint are non-commercial licensed models.
- No database or object store is used; intermediate files stay under the output folder.
- Object appearance uses multi-view vertex colors, not a UV texture atlas.
- Unseen object surfaces inherit color from the closest directly colored vertex.
