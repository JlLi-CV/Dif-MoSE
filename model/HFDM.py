import torch
import torch.nn as nn


class HLFAtt(nn.Module):

    def __init__(self, dim=4, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2,
                 alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim
        self.H, self.W = 64, 64


        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim


        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.ws = window_size


        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5


        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.lfsma_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.lfsma_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.lfsma_proj = nn.Linear(self.l_dim, self.l_dim)


        if self.h_heads > 0:
            self.hfdma_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.hfdma_proj = nn.Linear(self.h_dim, self.h_dim)

        self.final = nn.Sequential(nn.Conv2d(self.dim , self.dim , kernel_size=1),
                                   nn.LeakyReLU()
                                   )


    def hfdma(self, x):

        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)


        qkv = self.hfdma_qkv(x).reshape(
            B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads
        ).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3).reshape(
            B, h_group, w_group, self.ws, self.ws, self.h_dim
        )

        x = x.transpose(2, 3).reshape(B, H, W, self.h_dim)
        x = self.hfdma_proj(x)
        return x

    def lfsma(self, x):

        B, H, W, C = x.shape


        q = self.lfsma_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)


        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            kv = self.lfsma_kv(x_).reshape(
                B, -1, 2, self.l_heads, self.l_dim // self.l_heads
            ).permute(2, 0, 3, 1, 4)
        else:
            kv = self.lfsma_kv(x).reshape(
                B, -1, 2, self.l_heads, self.l_dim // self.l_heads
            ).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.lfsma_proj(x)
        return x

    def mlp(self , x):
        out = self.final(x)
        return out



    def forward(self, x):

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)


        if self.h_heads == 0:
            output = self.lfsma(x)
        elif self.l_heads == 0:

            output = self.hfdma(x)
        else:
            hfdma_out = self.hfdma(x)
            lfsma_out = self.lfsma(x)

            output = torch.cat((hfdma_out, lfsma_out), dim=-1)

        output = output.permute(0, 3, 1, 2)
        output = self.mlp(output)

        return output

if __name__ == '__main__':

    model = HLFAtt(dim=32)


    x = torch.randn(4, 32, 32, 32)

    output = model(x)



