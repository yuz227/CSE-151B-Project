import torch
import torch.nn as nn
import torch.nn.init as init
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)

    if cfg.model.type == "vision_transformer":
        return VisionTransformer(**model_kwargs)
    elif cfg.model.type == "improved_vit":
        return ImprovedVisionTransformer(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
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
        self.patch_embed = nn.Conv2d(
            in_channels=n_input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.h_patches = img_height // patch_size
        self.w_patches = img_width  // patch_size
        num_patches = self.h_patches * self.w_patches

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout_rate)

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
        init.trunc_normal_(self.pos_embed, std=0.02)
        init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            init.zeros_(self.patch_embed.bias)
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
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.head(x)
        x = x.view(
            B,
            self.h_patches,
            self.w_patches,
            self.n_output_channels,
            self.patch_size,
            self.patch_size
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(
            B,
            self.n_output_channels,
            self.h_patches * self.patch_size,
            self.w_patches * self.patch_size
        )
        return x


class ImprovedVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
