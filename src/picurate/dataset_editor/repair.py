"""Lazy mask (Hi-SAM, SAM-TS, SAM 2.1) and inpainting (Big-LaMa) adapters for watermark repair.

Weights come from the shared Hugging Face cache at pinned revisions. Hi-SAM and SAM 2 are installed from source by
`python -m picurate.dataset_editor.install_repair`; Big-LaMa only needs torch.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

MODEL_DEFAULTS = {
    "mask_backend": "Hi-SAM",
    "mask_device": "Auto",
    "lama_device": "Auto",
    "expansion": 3,
    "context": 128,
    "offline": False,
}
HF_TEXT = ("GoGiants1/Hi-SAM", "9fe50f111d0f1a3cce2ef6c377423f2eef6cfe69")
HF_SAM2 = ("facebook/sam2.1-hiera-small", "ee5bba1d82bb8749febdf90f45e84b687142ba03")
HF_LAMA = ("gaspardbruno/BigLama", "2bda1164365cebd245b0ae9ba23763f0f140d76c")


class InferenceEngine:
    def __init__(self):
        self.mask_model = None
        self.mask_key = None
        self.lama_model = None
        self.lama_key = None

    def device(self, choice):
        import torch

        if choice == "Auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if choice == "CPU":
            return "cpu"
        if not choice.startswith("cuda:") or not torch.cuda.is_available():
            raise ValueError(f"Requested device is unavailable: {choice}")
        torch.cuda.get_device_properties(torch.device(choice))
        return choice

    def download(self, repository, filename, offline):
        from huggingface_hub import hf_hub_download

        repo, revision = repository
        path = hf_hub_download(repo, filename, revision=revision, local_files_only=offline)
        return path, {"repo_id": repo, "revision": revision, "filename": filename}

    def segment(self, image, settings, box=None, points=None, labels=None):
        import numpy as np
        import torch

        backend = settings["mask_backend"]
        device = self.device(settings["mask_device"])
        key = (backend, device)
        if self.mask_key != key:
            self.mask_model = None
            self.mask_key = None
            if backend in ("Hi-SAM", "SAM-TS"):
                try:
                    from hi_sam.modeling.build import model_registry
                    from hi_sam.modeling.predictor import SamPredictor
                except ImportError as error:
                    raise RuntimeError("Hi-SAM is not installed. Run: python -m picurate.dataset_editor.install_repair") from error
                filename = "hi_sam_b.pth" if backend == "Hi-SAM" else "sam_tss_b_textseg.pth"
                path, spec = self.download(HF_TEXT, filename, settings["offline"])
                backbone, backbone_spec = self.download(HF_TEXT, "sam_vit_b_01ec64.pth", settings["offline"])
                args = SimpleNamespace(checkpoint=None, model_type="vit_b", hier_det=False, attn_layers=1, prompt_len=12)
                model = model_registry["vit_b"](args)
                state = torch.load(path, map_location="cpu", weights_only=True)
                sam_state = torch.load(backbone, map_location="cpu", weights_only=True)
                for name, value in sam_state.items():
                    state.setdefault(name, value)
                info = model.load_state_dict(state, strict=False)
                if info.missing_keys or any(not name.startswith("hi_decoder.") for name in info.unexpected_keys):
                    raise ValueError(f"Incompatible text model checkpoint: {info}")
                self.mask_model = SamPredictor(model.eval().to(device))
                self.mask_spec = [spec, backbone_spec]
            elif backend == "SAM 2.1":
                try:
                    from sam2.build_sam import build_sam2
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                except ImportError as error:
                    raise RuntimeError("SAM 2 is not installed. Run: python -m picurate.dataset_editor.install_repair") from error
                path, spec = self.download(HF_SAM2, "sam2.1_hiera_small.pt", settings["offline"])
                self.mask_model = SAM2ImagePredictor(
                    build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", path, device=device, apply_postprocessing=False)
                )
                self.mask_spec = [spec]
            else:
                raise ValueError(f"Unknown mask backend: {backend}")
            self.mask_key = key
        with torch.inference_mode():
            self.mask_model.set_image(np.asarray(image.convert("RGB")))
            if backend == "SAM 2.1":
                if box is None:
                    box = [0, 0, image.width - 1, image.height - 1]
                masks, scores, logits = self.mask_model.predict(
                    box=np.array(box),
                    point_coords=np.array(points) if points else None,
                    point_labels=np.array(labels) if points else None,
                    multimask_output=True,
                    return_logits=True,
                )
                index = int(np.argmax(scores))
                values = masks[index]
            else:
                _, masks, _, _ = self.mask_model.predict(multimask_output=False, return_logits=True)
                values = masks[0]
        return values.astype("float32"), {"backend": backend, "device": device, "models": self.mask_spec}

    def inpaint(self, image, mask, settings):
        import numpy as np
        import torch
        from PIL import Image

        device = self.device(settings["lama_device"])
        if self.lama_key != device:
            self.lama_model = None
            self.lama_key = None
            path, spec = self.download(HF_LAMA, "big-lama.pt", settings["offline"])
            with open(path, "rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9":
                raise ValueError("Big-LaMa checksum mismatch")
            self.lama_model = torch.jit.load(path, map_location=device).eval()
            self.lama_key = device
            self.lama_spec = {**spec, "sha256": checksum}
        rgb = np.array(image.convert("RGB"))
        selection = np.array(mask.convert("L")) > 0
        if not selection.any():
            raise ValueError("The mask is empty. Generate or paint a mask first.")
        h, w = selection.shape
        pad = ((0, (-h) % 8), (0, (-w) % 8))
        pixels = np.pad(rgb, (*pad, (0, 0)), mode="symmetric").transpose(2, 0, 1).copy()
        binary = np.pad(selection, pad, mode="symmetric").copy()
        with torch.inference_mode():
            tensor = torch.from_numpy(pixels).float().unsqueeze(0).to(device) / 255
            mask_tensor = torch.from_numpy(binary).float()[None, None].to(device)
            # No TorchScript fusion: fusing the graph (on its second call) from a Qt pool thread leaves that thread's
            # Python state broken, and the next model built in the pool aborts the interpreter.
            with torch.jit.fuser("none"):
                output = self.lama_model(tensor, mask_tensor)[0].permute(1, 2, 0).cpu().numpy()
        output = np.clip(output[:h, :w] * 255, 0, 255).astype("uint8")
        # Explicit compositing keeps every pixel outside the binary mask identical.
        rgb[selection] = output[selection]
        return Image.fromarray(rgb), {"backend": "Big-LaMa", "device": device, "model": self.lama_spec, "adapter_version": 1}
