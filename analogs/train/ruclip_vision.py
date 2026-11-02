"""Визуальный энкодер ruCLIP (Sber AI / ai-forever, Apache-2.0) без пакета `ruclip`.

Зачем: `pip install ruclip` тянет huggingface-hub==0.2.1 и youtokentome — на Windows
и Python 3.10+ это обычно не ставится. Для чертежей нужен только визуальный энкодер,
текстовая часть не используется. Архитектура ниже повторяет ruclip/model.py (v0.0.2),
имена весов совпадают с чекпойнтом (префикс "visual.").

Использование:
    from ruclip_vision import load_ruclip_visual
    model = load_ruclip_visual("models/ruclip-vit-large-patch14-336")  # папка с config.json и pytorch_model.bin
    emb = model(x)   # x: [B,3,336,336], нормализация CLIP_MEAN/CLIP_STD
"""
import json
import os
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

REPO_ID = "ai-forever/ruclip-vit-large-patch14-336"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads)
        self.ln_1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d, d * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(d * 4, d))]))
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.grad_checkpointing = False

    def forward(self, x):
        for blk in self.resblocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)          # NLD -> LND
        x = self.transformer(x).permute(1, 0, 2)    # LND -> NLD
        return self.ln_post(x[:, 0, :]) @ self.proj


def load_ruclip_visual(folder):
    cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
    width = cfg["vision_width"]
    model = VisionTransformer(cfg["image_resolution"], cfg["vision_patch_size"], width,
                              cfg["vision_layers"], width // 64, cfg["embed_dim"])
    sd = torch.load(os.path.join(folder, "pytorch_model.bin"), map_location="cpu")
    vis = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    model.load_state_dict(vis, strict=True)   # упадёт с понятной ошибкой, если веса не те
    return model.float()
