"""Panorama dense encoder: a frozen planar SSL backbone adapted with LoRA.

Rationale (memory/project-thesis.md): panorama data is too scarce to pretrain DINOv3-scale,
so we TRANSFER a frozen planar encoder and only learn a small LoRA adapter (Surgical-DINO
precedent: LoRA on attention Q,V, <3% params). The frozen teacher is recovered for free by
disabling the adapter, so distillation needs no second model copy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_tiles(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1] -> imagenet-normalized."""
    return (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)


class PanoEncoder(nn.Module):
    def __init__(self, model_id: str = "facebook/dinov2-base", lora_rank: int = 16,
                 lora_alpha: int = 32, lora_dropout: float = 0.0, freeze: bool = True,
                 adapter_path: str = None):
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

            if adapter_path is not None:                          # load a trained adapter for eval
                backbone = PeftModel.from_pretrained(backbone, adapter_path)
            else:
                # attention naming differs by backbone: DINOv2/BERT use query/value,
                # DINOv3 uses LLaMA-style q_proj/v_proj. Pick whichever exists.
                leaf = {n.split(".")[-1] for n, _ in backbone.named_modules()}
                if {"q_proj", "v_proj"} <= leaf:
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

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
