"""Reusable ShapeR inference runtime for one or more local PKL samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import omegaconf
import torch
import trimesh

from dataset.shaper_dataset import InferenceDataset
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from postprocessing.helper import remove_floating_geometry


PRESETS = {
    "quality": (16, 4, 50),
    "speed": (4, 2, 10),
    "balance": (16, 4, 25),
}


class ShapeRRuntime:
    """Load ShapeR once and reconstruct multiple object samples."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints", device: str = "cuda"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("ShapeR video inference requires a CUDA GPU")

        config_path = self.checkpoint_dir / "config.yaml"
        checkpoint_path = self.checkpoint_dir / "019-0-bfloat16.ckpt"
        vae_path = self.checkpoint_dir / "vae-088-0-bfloat16.ckpt"
        for path in (config_path, checkpoint_path, vae_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing ShapeR checkpoint: {path}")

        self.config = omegaconf.OmegaConf.load(config_path)
        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model = ShapeRDenoiser(self.config).to(self.device)
        self.model.convert_to_bfloat16()
        self.model.load_state_dict(state_dict, strict=False)
        self.model = torch.compile(self.model, fullgraph=True).eval()

        self.vae = MichelangeloLikeAutoencoderWrapper(str(vae_path), self.device)
        self.vae.model.use_udf_extraction = True
        self.vae.model.udf_iso = 0.375
        self.text_encoder = TextFeatureExtractor(device=self.device).to(torch.bfloat16)

    @torch.no_grad()
    def reconstruct_many(
        self,
        samples: list[str | Path],
        output_dir: str | Path,
        preset: str = "balance",
        transform_to_world: bool = True,
        simplify: bool = True,
        remove_floating: bool = True,
    ) -> list[Path]:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        if not samples:
            return []

        num_images, token_multiplier, num_steps = PRESETS[preset]
        sample_paths = [str(Path(path).resolve()) for path in samples]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = InferenceDataset(
            self.config, paths=sample_paths, override_num_views=num_images
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.custom_collate,
        )

        scales = self.vae.model.get_token_scales()
        scale_prob = np.zeros_like(scales)
        scale_prob[6] = 1.0
        self.vae.model.set_inference_scale_probabilities(scale_prob)
        token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
        token_shape = (1, token_count, self.vae.get_embed_dim())
        shifted = getattr(self.config.fm_transformer, "time_sampler", "lognorm") == "flux"

        results: list[Path] = []
        for batch in loader:
            batch = InferenceDataset.move_batch_to_device(
                batch, self.device, dtype=torch.bfloat16
            )
            latents = self.model.infer_latents(
                batch,
                token_shape=token_shape,
                text_feature_extractor=self.text_encoder,
                num_steps=num_steps,
                use_shifted_sampling=shifted,
            )
            mesh = self.vae.infer_mesh_from_latents(latents)[0]
            if remove_floating:
                mesh = remove_floating_geometry(mesh)
            if simplify and len(mesh.faces) > 125_000:
                mesh = mesh.simplify_quadric_decimation(face_count=125_000)
            mesh = dataset.rescale_back(
                int(batch["index"][0]), mesh, transform_to_world
            )
            output_path = output_dir / f"{batch['name'][0]}.glb"
            mesh.export(output_path, include_normals=True)
            results.append(output_path)
        return results

