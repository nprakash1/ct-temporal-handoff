"""
Shared helpers for embedding CT scans and text with a FROZEN CT-CLIP
(the CT-RATE-native chest-CT foundation model), as a drop-in alternative to
merlin_utils.MerlinEmbedder.

CT-CLIP (Hamamci et al., https://github.com/ibrahimethemhamamci/CT-CLIP):
  - image encoder : CTViT  (3D ViT)          -> flatten -> to_visual_latent -> 512-d
  - text  encoder : microsoft/BiomedVLP-CXR-BERT-specialized (768-d) -> to_text_latent -> 512-d
  - shared latent : 512-d, L2-normalized (same joint space as our MERLIN pipeline)

CT-CLIP preprocessing (DIFFERENT from MERLIN -- must match or features are garbage):
  - HU clip      : [-1000, +200]
  - normalize    : linearly to [-1, 1]
  - spacing      : 0.75 x 0.75 x 1.5 mm  (x, y, z)
  - target shape : 480 x 480 x 240        (H, W, D)  -> tensor (1, 1, D, H, W)

The CT-CLIP weights are never updated (everything runs under torch.no_grad()).

NOTE: the exact attribute names below (`visual_transformer`, `text_transformer`,
`to_visual_latent`, `to_text_latent`) follow CT_CLIP/ct_clip.py. The Colab notebook
has a one-cell VERIFY step that prints the model's attributes and a forward-shape
check before the full run, so any drift is caught immediately.
"""

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# CT-CLIP preprocessing constants (from arXiv:2403.17834 + repo)
HU_MIN, HU_MAX = -1000.0, 200.0
TARGET_SPACING = (0.75, 0.75, 1.5)      # x, y, z (mm)
TARGET_SHAPE_HWD = (480, 480, 240)      # H, W, D

REPO_ID = "ibrahimhamamci/CT-RATE"
CTCLIP_WEIGHTS_HF = "models/CT-CLIP-Related/CT-CLIP_v2.pt"
TEXT_MODEL_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _resample_to_spacing(vol: np.ndarray, spacing_xyz, target_xyz):
    """Resample a (H, W, D) volume from `spacing_xyz` to `target_xyz` (mm) via
    linear interpolation. spacing/target are (x, y, z)."""
    from scipy.ndimage import zoom
    sx, sy, sz = spacing_xyz
    tx, ty, tz = target_xyz
    # array axes are (H=y, W=x, D=z); map spacings accordingly
    factors = (sy / ty, sx / tx, sz / tz)
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return vol
    return zoom(vol, factors, order=1)


def _center_crop_pad(vol: np.ndarray, target_hwd, pad_value: float):
    """Center crop/pad a (H, W, D) array to exactly target_hwd."""
    out = np.full(target_hwd, pad_value, dtype=vol.dtype)
    src_slices, dst_slices = [], []
    for ax in range(3):
        s, t = vol.shape[ax], target_hwd[ax]
        if s >= t:  # crop
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:       # pad (center)
            start = (t - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(start, start + s))
    out[tuple(dst_slices)] = vol[tuple(src_slices)]
    return out


def preprocess_ct(path: str) -> torch.Tensor:
    """Load a NIfTI CT and return a CT-CLIP-ready tensor of shape (1, 1, D, H, W).

    Steps: read HU + spacing -> clip [-1000,200] -> normalize [-1,1]
           -> resample to 0.75/0.75/1.5 mm -> center crop/pad to 480x480x240.
    """
    import nibabel as nib

    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)          # (H, W, D)
    zooms = img.header.get_zooms()[:3]                # (x, y, z) mm

    # HU clip + normalize to [-1, 1]
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)          # -> [0, 1]
    vol = vol * 2.0 - 1.0                             # -> [-1, 1]

    # resample to target spacing, then fix to target shape (pad with air = -1)
    vol = _resample_to_spacing(vol, zooms, TARGET_SPACING)
    vol = _center_crop_pad(vol, TARGET_SHAPE_HWD, pad_value=-1.0)

    # (H, W, D) -> (D, H, W) -> (1, 1, D, H, W)
    vol = np.transpose(vol, (2, 0, 1))
    t = torch.from_numpy(np.ascontiguousarray(vol)).float()
    return t.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class CTCLIPEmbedder:
    """Frozen CT-CLIP -> 512-d image and text embeddings (same joint space).

    Parallels merlin_utils.MerlinEmbedder so downstream callers are unchanged.
    Requires the CT-CLIP repo installed:
        pip install -e transformer_maskgit && pip install -e CT_CLIP
    """

    def __init__(self, weights_path: str, device: Optional[str] = None):
        from transformers import BertModel, BertTokenizer
        try:
            from ct_clip import CTCLIP           # from CT_CLIP package
            from transformer_maskgit import CTViT
        except Exception as e:  # pragma: no cover - import guidance
            raise ImportError(
                "CT-CLIP not installed. In the repo root run:\n"
                "  cd transformer_maskgit && pip install -e . && cd ..\n"
                "  cd CT_CLIP && pip install -e . && cd ..\n"
                f"(original error: {e})"
            )

        self.device = device or get_device()

        # text tower: CT-CLIP uses a PLAIN transformers BertModel over the CXR-BERT
        # weights (NOT the custom CXRBert class, which nests params under `.bert`
        # and breaks the checkpoint key names).
        self.tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_ID, do_lower_case=True)
        text_encoder = BertModel.from_pretrained(TEXT_MODEL_ID)

        # image tower (CTViT) -- exact config from run_zero_shot.py
        image_encoder = CTViT(
            dim=512, codebook_size=8192, image_size=480,
            patch_size=20, temporal_patch_size=10,
            spatial_depth=4, temporal_depth=4, dim_head=32, heads=8,
        )

        self.clip = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912, dim_text=768, dim_latent=512,
            extra_latent_projection=False, use_mlm=False,
            downsample_image_embeds=False, use_all_token_embeds=False,
        )
        # load pretrained weights tolerantly: newer transformers drops the
        # `embeddings.position_ids` buffer, so strict=True would fail on that one
        # benign key. We load strict=False and assert nothing important is missing.
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and "state_dict" not in sd \
                and all(isinstance(v, dict) for v in [sd.get("model", {})]):
            sd = sd["model"]
        missing, unexpected = self.clip.load_state_dict(sd, strict=False)
        benign = {"text_transformer.embeddings.position_ids"}
        real_missing = [k for k in missing]
        real_unexpected = [k for k in unexpected if k not in benign]
        if real_missing or real_unexpected:
            print(f"[CTCLIPEmbedder] load_state_dict mismatches "
                  f"(missing={len(real_missing)}, unexpected={len(real_unexpected)})")
            if real_missing:
                print("  missing[:5]   :", real_missing[:5])
            if real_unexpected:
                print("  unexpected[:5]:", real_unexpected[:5])
        else:
            print("[CTCLIPEmbedder] weights loaded cleanly "
                  f"(ignored benign: {sorted(benign & set(unexpected))})")
        self.clip.eval().to(self.device)
        for p in self.clip.parameters():
            p.requires_grad_(False)

    # ---- Images --------------------------------------------------------
    @torch.no_grad()
    def embed_image_tensor(self, vol: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """vol: (1,1,D,H,W) -> (1,512)."""
        vol = vol.to(self.device)
        # CTViT returns encoded tokens of shape (b, h, w, z, c) = (1,24,24,24,512).
        # CT-CLIP's forward reduces this with mean over dim=1, THEN flattens to
        # (b, 24*24*512 = 294912) before to_visual_latent (see ct_clip.py L719-740).
        tokens = self.clip.visual_transformer(vol, return_encoded_tokens=True)
        tokens = torch.mean(tokens, dim=1)                  # (1, 24, 24, 512)
        feat = tokens.reshape(tokens.shape[0], -1)          # flatten -> (1, 294912)
        latent = self.clip.to_visual_latent(feat)           # -> (1, 512)
        out = latent.detach().cpu().float()
        return F.normalize(out, dim=-1) if normalize else out

    @torch.no_grad()
    def embed_image_path(self, path: str, normalize: bool = True) -> torch.Tensor:
        return self.embed_image_tensor(preprocess_ct(path), normalize=normalize)

    # ---- Text ----------------------------------------------------------
    @torch.no_grad()
    def embed_texts(self, texts: List[str], normalize: bool = True,
                    max_length: int = 512) -> torch.Tensor:
        """list[str] -> (N, 512) using the CXR-BERT tower + to_text_latent (CLS)."""
        enc = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        out = self.clip.text_transformer(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        )
        cls = out[0][:, 0, :]                                # (N, 768) CLS
        latent = self.clip.to_text_latent(cls)              # (N, 512)
        out = latent.detach().cpu().float()
        return F.normalize(out, dim=-1) if normalize else out


# ---------------------------------------------------------------------------
# Volume download (reused from scripts/14) -- stream one at a time
# ---------------------------------------------------------------------------
def remote_candidates(vol: str):
    """HF paths for a CT-RATE volume like 'train_3_b_1.nii.gz'."""
    base = vol.replace(".nii.gz", "").replace(".nii", "")
    split, pid, scan = base.split("_")[:3]
    patient = f"{split}_{pid}"
    scan_folder = f"{split}_{pid}_{scan}"
    split_folders = ([f"{split}_fixed", split] if split in ("train", "valid") else [split])
    return [f"dataset/{sf}/{patient}/{scan_folder}/{vol}" for sf in split_folders]


def download_volume(vol: str, token: Optional[str], out_dir: str):
    from huggingface_hub import hf_hub_download
    for path in remote_candidates(vol):
        try:
            return hf_hub_download(REPO_ID, path, repo_type="dataset",
                                   token=token, local_dir=out_dir)
        except Exception:
            continue
    return None
