# src/models.py

import torch
import torch.nn as nn
import torch.nn.init as init
from omegaconf import DictConfig


def get_model(cfg: DictConfig) -> nn.Module:
    """
    根据 cfg.model.type 实例化 Base ViT，然后用 NormalizedModel 包装，
    在 forward 中自动做通道归一化/反归一化。
    需要在 config.yaml 中添加以下字段：
      model.input_mean:   [co2_mean, so2_mean, ch4_mean, bc_mean, rsdt_mean]
      model.input_std:    [co2_std,  so2_std,  ch4_std,  bc_std,  rsdt_std]
      model.output_mean:  [tas_mean, pr_mean]
      model.output_std:   [tas_std,  pr_std]
    """
    # 提取 ViT 需要的超参（除去 type 和归一化统计量）
    base_kwargs = {
        k: v for k, v in cfg.model.items()
        if k not in ("type", "input_mean", "input_std", "output_mean", "output_std")
    }
    base_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    base_kwargs["n_output_channels"] = len(cfg.data.output_vars)

    # 选 Base ViT
    if cfg.model.type == "vision_transformer":
        base_model = VisionTransformer(**base_kwargs)
    elif cfg.model.type == "improved_vit":
        base_model = ImprovedVisionTransformer(**base_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # 用 NormalizedModel 包装
    return NormalizedModel(
        base_model,
        input_mean=cfg.model.input_mean,
        input_std=cfg.model.input_std,
        output_mean=cfg.model.output_mean,
        output_std=cfg.model.output_std,
    )


class NormalizedModel(nn.Module):
    """
    Wrapper: 在 forward 中对 x 做 (x - mean) / std，再把 base_model 的输出反归一化。
    """
    def __init__(
        self,
        base_model: nn.Module,
        input_mean: list[float],
        input_std:  list[float],
        output_mean: list[float],
        output_std:  list[float],
    ):
        super().__init__()
        self.base = base_model
        # 注册 buffer 方便广播
        self.register_buffer("in_mean",  torch.tensor(input_mean).view(1, -1, 1, 1))
        self.register_buffer("in_std",   torch.tensor(input_std).view(1, -1, 1, 1))
        self.register_buffer("out_mean", torch.tensor(output_mean).view(1, -1, 1, 1))
        self.register_buffer("out_std",  torch.tensor(output_std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准化输入
        x_norm = (x - self.in_mean) / self.in_std
        # base_model 输出也是标准化尺度
        y_norm = self.base(x_norm)
        # 反归一化到物理量
        return y_norm * self.out_std + self.out_mean


class PatchEmbed(nn.Module):
    """把图像分块并做线性投影"""
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)                     # -> (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)     # -> (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """
    官方风格的 ViT：TransformerEncoderLayer + 合理初始化 + Head Dropout/LayerNorm
    """
    def __init__(
        self,
        n_input_channels: int,
        n_output_channels: int,
        img_height: int,
        img_width: int,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=n_input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # 网格尺寸
        self.h_patches = img_height // patch_size
        self.w_patches = img_width  // patch_size
        num_patches = self.h_patches * self.w_patches

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout_rate)

        # 官方 TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True,
            layer_norm_eps=1e-5,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Head: LayerNorm + Dropout + 2-layer MLP
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, patch_size * patch_size * n_output_channels),
        )

        self.n_output_channels = n_output_channels
        self.patch_size = patch_size

        self._init_weights()

    def _init_weights(self):
        # Conv & pos_embed 初始化
        init.trunc_normal_(self.pos_embed, std=0.02)
        init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            init.zeros_(self.patch_embed.bias)
        # head 中的 Linear/LayerNorm 初始化
        for m in self.head:
            if isinstance(m, nn.Linear):
                init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # Patch embedding
        x = self.patch_embed(x)                 # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)        # (B, num_patches, embed_dim)
        # 加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        # Transformer 编码
        x = self.transformer(x)
        # Head
        x = self.head(x)                        # (B, num_patches, C_out*ps*ps)
        # 重塑回 (B, C_out, H, W)
        x = x.view(
            B,
            self.h_patches,
            self.w_patches,
            self.n_output_channels,
            self.patch_size,
            self.patch_size
        ).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(
            B,
            self.n_output_channels,
            self.h_patches * self.patch_size,
            self.w_patches * self.patch_size
        )
        return x


class ImprovedVisionTransformer(VisionTransformer):
    """
    改进版 ViT：可在此加入 DropPath、LayerScale 等提分技巧，
    保留与 VisionTransformer 相同的接口 & forward 行为。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在这里可以插入你的 Stochastic Depth / LayerScale 等改进
        # 例如：self.drop_path = DropPath(...)

    # 如果只继承不改 forward，则无需重写 forward
