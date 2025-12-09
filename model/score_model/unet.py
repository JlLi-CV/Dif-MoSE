from typing import List, Optional, Set, Tuple, Union, Callable
import math
import torch
import torch as th
from einops import rearrange
from torch import nn


def conv_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def normalization(channels: int, dims: int = 2):
    if dims == 2:
        return nn.BatchNorm2d(channels)
    elif dims == 1:
        return nn.LayerNorm(channels)
    raise ValueError(f"unsupported dimensions for normalization: {dims}")


def timestep_embedding(timesteps: torch.Tensor, dim: int, repeat_only: bool = False):
    if not repeat_only:
        half = dim // 2
        emb = torch.exp(
            torch.arange(half, dtype=torch.float32, device=timesteps.device) * (-math.log(10000) / (half - 1)))
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    else:
        emb = torch.ones_like(timesteps[:, None])
        emb = emb.repeat(1, dim)
    return emb


def zero_module(module: nn.Module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class TimestepBlock(nn.Module):
    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: Optional[torch.Tensor] = None):
        raise NotImplementedError()


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        if self.use_conv:
            x = self.conv(x)
        return x


def checkpoint(func: Callable, inputs: Tuple, params: List, use_checkpoint: bool = False):
    if use_checkpoint:
        return nn.utils.checkpoint.checkpoint(func, *inputs)
    else:
        return func(*inputs)


class VanillaAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape

        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(2, 3).reshape(b, c, h, w)
        out = self.proj(out)
        return out + x


def make_attn(channels: int, attn_type: str = "vanilla") -> nn.Module:
    if attn_type == "vanilla":
        return VanillaAttention(channels)
    else:
        raise ValueError(f"unsupported attention type: {attn_type}")


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            dropout: float = 0.0,
            attn_type: str = "vanilla",
            time_embed_dim: Optional[int] = None,
            use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm


        self.temb_proj = None
        if time_embed_dim is not None and time_embed_dim > 0:
            if use_scale_shift_norm:
                self.temb_proj = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, out_channels * 2)
                )
            else:
                self.temb_proj = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, out_channels)
                )


        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)


        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)


        self.attn = make_attn(out_channels, attn_type=attn_type) if attn_type != "none" else nn.Identity()

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, temb=None):

        x_orig = x


        h = self.norm1(x)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            temb_out = self.temb_proj(temb)

            if self.use_scale_shift_norm:
                scale, shift = temb_out.chunk(2, dim=1)

                scale = scale.unsqueeze(-1).unsqueeze(-1)
                shift = shift.unsqueeze(-1).unsqueeze(-1)
                h = h * (1 + scale) + shift
            else:

                h = h + temb_out.unsqueeze(-1).unsqueeze(-1)

        h = self.relu1(h)

        h = self.norm2(h)
        h = self.conv2(h)
        h = self.relu2(h)

        h = self.attn(h)

        h = self.dropout(h)
        h = self.conv3(h)

        x_shortcut = self.shortcut(x_orig)

        return x_shortcut + h


class AttentionBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            num_heads: Optional[int] = 1,
            num_head_channels: Optional[int] = -1,
            use_checkpoint: Optional[bool] = False,
            use_new_attention_order: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint

        self.norm = nn.LayerNorm(channels)
        self.attn = make_attn(channels)
        self.proj_out = zero_module(conv_nd(2, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x_orig = x

        x_flat = x.reshape(b, c, h * w).permute(0, 2, 1)
        x_norm = self.norm(x_flat).permute(0, 2, 1).reshape(b, c, h, w)

        h_attn = self.attn(x_norm)

        h = self.proj_out(h_attn)

        return x_orig + h


class SpatialTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_heads: int,
            d_head: int,
            depth: Optional[int] = 1,
            dropout: Optional[float] = 0.0,
            context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.BatchNorm2d(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        class TransformerBlock(nn.Module):
            def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn1 = VanillaAttention(dim, n_heads)
                self.norm2 = nn.LayerNorm(dim)
                self.ff = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                )

            def forward(self, x, context=None):
                b, t, c = x.shape
                h = w = int(math.sqrt(t))
                x_4d = x.permute(0, 2, 1).reshape(b, c, h, w)

                attn_out = self.attn1(self.norm1(x).permute(0, 2, 1).reshape(b, c, h, w))
                attn_out = attn_out.reshape(b, c, t).permute(0, 2, 1)
                x = x + attn_out

                x = x + self.ff(self.norm2(x))
                return x

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(inner_dim, n_heads, d_head, dropout, context_dim) for _ in range(depth)]
        )
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(
            self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        for block in self.transformer_blocks:
            x = block(x, context)

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(
            self, x: torch.Tensor, emb: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResnetBlock):

                x = layer(x, temb=emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            model_channels: int,
            out_channels: int,
            num_res_blocks: int,
            attention_resolutions: Union[Set[int], List[int], Tuple[int]],
            dropout: float = 0,
            channel_mult: Tuple[int] = (1, 2, 4, 8),
            conv_resample: bool = True,
            dims: int = 2,
            num_classes: Optional[int] = None,
            use_checkpoint: bool = False,
            use_fp16: bool = False,
            num_heads: int = -1,
            num_head_channels: int = -1,
            num_heads_upsample: int = -1,
            use_scale_shift_norm: bool = False,
            resblock_updown: bool = False,
            use_new_attention_order: bool = False,
            use_spatial_transformer: bool = False,
            transformer_depth: int = 1,
            context_dim: Optional[int] = None,
            n_embed: Optional[int] = None,
            legacy: bool = True,
            low_res_context_channels: Optional[int] = None,
            **ignorekwargs: dict,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.predict_codebook_ids = n_embed is not None

        self.low_res_context_channels = low_res_context_channels


        self.init_fea = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


        self.low_res_context_proj = None
        if low_res_context_channels is not None:

            self.low_res_context_proj = nn.Conv2d(low_res_context_channels, in_channels, kernel_size=1)

            self.input_blocks = nn.ModuleList([
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels * 2, model_channels, 3, padding=1)
                )
            ])
        else:

            self.input_blocks = nn.ModuleList([
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        attn_type="vanilla",
                        time_embed_dim=time_embed_dim if not use_spatial_transformer else None,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            num_head_channels if num_head_channels != -1 else ch // num_heads,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            # 下采样块
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResnetBlock(
                            in_channels=ch,
                            out_channels=out_ch,
                            dropout=dropout,
                            time_embed_dim=time_embed_dim if not use_spatial_transformer else None,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResnetBlock(
                in_channels=ch,
                out_channels=ch,
                dropout=dropout,
                time_embed_dim=time_embed_dim if not use_spatial_transformer else None,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                num_head_channels if num_head_channels != -1 else ch // num_heads,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResnetBlock(
                in_channels=ch,
                out_channels=ch,
                dropout=dropout,
                time_embed_dim=time_embed_dim if not use_spatial_transformer else None,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()

                layers = [
                    ResnetBlock(
                        in_channels=ch + ich,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        time_embed_dim=time_embed_dim if not use_spatial_transformer else None,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult

                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            num_head_channels if num_head_channels != -1 else ch // num_heads,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )

                # 上采样块
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResnetBlock(
                            in_channels=ch,
                            out_channels=out_ch,
                            dropout=dropout,
                            time_embed_dim=time_embed_dim if not use_spatial_transformer else None,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch), conv_nd(dims, model_channels, n_embed, 1)
            )

    def convert_to_fp16(self):
        self.input_blocks.apply(lambda m: m.half() if isinstance(m, nn.Module) and not isinstance(m, (
            nn.BatchNorm2d, nn.Dropout, nn.LayerNorm)) else m)
        self.middle_block.apply(lambda m: m.half() if isinstance(m, nn.Module) and not isinstance(m, (
            nn.BatchNorm2d, nn.Dropout, nn.LayerNorm)) else m)
        self.output_blocks.apply(lambda m: m.half() if isinstance(m, nn.Module) and not isinstance(m, (
            nn.BatchNorm2d, nn.Dropout, nn.LayerNorm)) else m)

    def convert_to_fp32(self):
        self.input_blocks.apply(lambda m: m.float() if isinstance(m, nn.Module) else m)
        self.middle_block.apply(lambda m: m.float() if isinstance(m, nn.Module) else m)
        self.output_blocks.apply(lambda m: m.float() if isinstance(m, nn.Module) else m)

    def forward(self, x, timesteps=None, context=None, y=None, low_res_context=None, **kwargs):

        assert (y is not None) == (
                self.num_classes is not None), "must specify y if and only if the model is class-conditional"

        hs = []


        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)


        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)


        if low_res_context is not None and self.low_res_context_proj is not None:
            context_proj = self.low_res_context_proj(low_res_context)

            h = torch.cat([x, context_proj], dim=1)
        else:
            h = x
        h = h.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)


        h = self.middle_block(h, emb, context)


        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


if __name__ == '__main__':

    config = {
        "in_channels": 64,
        "model_channels": 160,
        "out_channels": 64,
        "num_res_blocks": 2,
        "attention_resolutions": [8, 16],
        "dropout": 0.1,
        "channel_mult": [1, 2, 4, 4],
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "use_fp16": False,
        "num_heads": 8,
        "num_head_channels": 32,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "use_spatial_transformer": False,
        "transformer_depth": 1,
        "context_dim": None,
        "n_embed": None,
        "legacy": True,
        "low_res_context_channels": 64,
    }

    model = UNetModel(**config)

