"""Panorama dense encoder: a frozen planar SSL backbone adapted with LoRA.

Rationale (memory/project-thesis.md): panorama data is too scarce to pretrain DINOv3-scale,
so we TRANSFER a frozen planar encoder and only learn a small LoRA adapter (Surgical-DINO
precedent: LoRA on attention Q,V, <3% params). The frozen teacher is recovered for free by
disabling the adapter, so distillation needs no second model copy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_tiles(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1] -> imagenet-normalized."""
    return (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)


class PanoEncoder(nn.Module):
    def __init__(self, model_id: str = "facebook/dinov2-base", lora_rank: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.0, freeze: bool = True,
                 adapter_path: str = None, adapter_trainable: bool = False,
                 lora_targets: list = None):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_id)
        self.patch: int = backbone.config.patch_size
        self.dim: int = backbone.config.hidden_size
        self.model_id = model_id

        for p in backbone.parameters():
            p.requires_grad = False

        self.lora = lora_rank > 0 or adapter_path is not None
        if self.lora:
            from peft import LoraConfig, get_peft_model, PeftModel

            if adapter_path is not None:                          # load a trained adapter (eval, or
                backbone = PeftModel.from_pretrained(              # continued training when trainable)
                    backbone, adapter_path, is_trainable=adapter_trainable)
            else:
                # attention naming differs by backbone: DINOv2/BERT use query/value,
                # DINOv3 uses LLaMA-style q_proj/v_proj. Pick whichever exists — unless the caller
                # passes explicit lora_targets (e.g. all attn+MLP linears to widen adaptation).
                leaf = {n.split(".")[-1] for n, _ in backbone.named_modules()}
                if lora_targets:
                    missing = [t for t in lora_targets if t not in leaf]
                    if missing:
                        raise ValueError(f"lora_targets {missing} absent in {model_id}; leaves={sorted(leaf)[:20]}")
                    targets = list(lora_targets)
                elif {"q_proj", "v_proj"} <= leaf:
                    targets = ["q_proj", "v_proj"]
                elif {"query", "value"} <= leaf:
                    targets = ["query", "value"]
                else:
                    raise ValueError(f"no known attn q/v modules in {model_id}; leaves={sorted(leaf)[:20]}")
                cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 target_modules=targets, bias="none")
                backbone = get_peft_model(backbone, cfg)
        self.backbone = backbone

    def _dense(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) normalized -> (B, D, Gh, Gw). Robust to CLS/register prefix tokens."""
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        try:
            out = self.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        except TypeError:
            out = self.backbone(pixel_values=x).last_hidden_state       # (B, prefix+gh*gw, D)
        patches = out[:, out.shape[1] - gh * gw:, :]                    # patch tokens are last
        return patches.transpose(1, 2).reshape(b, self.dim, gh, gw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Student (adapted) dense features."""
        return self._dense(x)

    @torch.no_grad()
    def teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen-teacher dense features (LoRA disabled). Detached."""
        was_training = self.training
        self.eval()
        if self.lora:
            with self.backbone.disable_adapter():
                feat = self._dense(x)
        else:
            feat = self._dense(x)
        if was_training:
            self.train()
        return feat

    def forward_masked(self, x: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
        """iBOT masked forward: masked patches (bool_masked_pos, (B, Gh*Gw)) are replaced by the
        backbone's mask_token before the transformer, so the encoder never sees masked content
        (no leakage). Returns (B, D, Gh, Gw) student features. DINOv3ViTModel supports this natively."""
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        try:
            out = self.backbone(pixel_values=x, bool_masked_pos=bool_masked_pos,
                                interpolate_pos_encoding=True).last_hidden_state
        except TypeError:
            out = self.backbone(pixel_values=x, bool_masked_pos=bool_masked_pos).last_hidden_state
        patches = out[:, out.shape[1] - gh * gw:, :]
        return patches.transpose(1, 2).reshape(b, self.dim, gh, gw)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """EMA-update the teacher's adapter from the student, matched by parameter name.

    Only parameters trainable on the student side (the LoRA adapter) are updated;
    the shared frozen backbone stays identical. F-3's 'slowly evolving anchor'."""
    t_params = dict(teacher.named_parameters())
    for name, p in student.named_parameters():
        if p.requires_grad:
            t_params[name].data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)


class Expander(nn.Module):
    """VICReg expander: dense features (B,D,Gh,Gw) -> (B,P,Gh,Gw) per patch.

    Canonical VICReg puts variance/covariance on an EXPANDER, not the raw backbone —
    which is exactly the fix for F-3's neutered anti-collapse (docs §11.2). Discarded
    downstream; its only job is to give var/cov a well-scaled space to act in.
    """

    def __init__(self, dim: int, proj_dim: int = 1024, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, proj_dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, d, gh, gw = feat.shape
        x = self.net(feat.permute(0, 2, 3, 1).reshape(-1, d))
        return x.reshape(b, gh, gw, -1).permute(0, 3, 1, 2)


class GlobalExpander(nn.Module):
    """Project one spatially pooled descriptor per tile for global VICReg.

    Tile descriptors supply many more independent samples than one pooled descriptor per
    panorama while the paired photometric views retain an exact one-to-one target.
    """

    def __init__(self, dim: int, proj_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, proj_dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat.mean(dim=(2, 3)))


class SubtokenExpander(nn.Module):
    """Learn a 2x dense auxiliary grid from patch tokens for sub-token VICReg.

    This head is discarded downstream. It does not pretend the ViT has native sub-tokens;
    it creates a supervised high-resolution interpolation path whose gradients still reach
    the adapted backbone.
    """

    def __init__(self, dim: int, proj_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GroupNorm(16, hidden), nn.GELU())
        self.post = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.GELU(),
            nn.Conv2d(hidden, proj_dim, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(self.pre(feat), scale_factor=2, mode="bilinear", align_corners=False)
        return self.post(x)


class GeometryHead(nn.Module):
    """Auxiliary readout that keeps latitude and tile-relative direction decodable."""

    def __init__(self, dim: int, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, 256, 1), nn.GELU(), nn.Conv2d(256, out_dim, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class CrossViewPredictor(nn.Module):
    """Term B (docs/PANO_WHEREWHAT_SPEC.md §3): predict a masked A-patch's feature from the
    overlapping B-tile's evidence at the warp location + A's own (masked) context.

    Residual on the B-evidence with a ZERO-INIT output layer, so at step 0 pred == b_ev exactly
    = the naive cross-view baseline (F-2's zero-init trick: training can only add on top). The
    target is A's FROZEN feature (de-overlap rule) so the net must learn the B->A distortion
    transform, not copy — while the frozen anchor keeps it non-erosive.
    """

    def __init__(self, dim: int, hidden: int = None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, a_ctx: torch.Tensor, b_ev: torch.Tensor) -> torch.Tensor:
        """a_ctx, b_ev: (n, D). Returns (n, D) predicted A feature."""
        return b_ev + self.net(torch.cat([a_ctx, b_ev], dim=-1))


class CodeHead(nn.Module):
    """M1 semantic-code head (SI-SSL §2.1): projector + K normalized prototypes.

    Maps dense backbone features (B, D, Gh, Gw) -> prototype scores (B, K, Gh, Gw).
    Scores are cosine(z, c_k) in [-1, 1]; the temperature lives in the loss. The head
    is discarded downstream — its only job is to press semantic identity into F.
    """

    def __init__(self, dim: int, proj_dim: int = 256, n_proto: int = 512):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, proj_dim))
        self.prototypes = nn.Linear(proj_dim, n_proto, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        """SwAV-style: keep prototype vectors on the unit sphere (call once per step)."""
        self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = self.proj(feat.permute(0, 2, 3, 1))            # (B, Gh, Gw, P)
        z = F.normalize(z, dim=-1)
        return self.prototypes(z).permute(0, 3, 1, 2)      # (B, K, Gh, Gw)
