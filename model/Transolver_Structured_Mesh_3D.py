import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
import torch.utils.checkpoint as checkpoint
from einops import rearrange

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Project(nn.Module):
    def __init__(self, dim, inner_dim, D, H, W, inputBeam_num, kernel):
        super().__init__()
        self.D = D
        self.H = H
        self.W = W
        self.inner_dim = inner_dim
        self.inputBeam_num = inputBeam_num
        self.beam_project = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.elas_project = nn.Linear(dim, inner_dim)

    def forward(self, x):
        B, N, C = x.shape
        x_beam = self.beam_project(x[:, :self.inputBeam_num, :].reshape(B, self.D, self.H, self.W, C).contiguous().permute(0, 4, 1, 2, 3).contiguous())
        x_elas = self.elas_project(x[:, self.inputBeam_num:, :])
        x_beam = x_beam.permute(0, 2, 3, 4, 1).contiguous().reshape(B, self.inputBeam_num, self.inner_dim).contiguous()
        x = torch.cat([x_beam, x_elas], dim=1)
        return x

class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=16, dropout=0., slice_num=64, D=51, H=6, W=6, inputBeam_num=1836, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.D = D
        self.H = H
        self.W = W

        self.in_project_x = Project(dim, inner_dim, D, H, W, inputBeam_num, kernel)
        self.in_project_fx = Project(dim, inner_dim, D, H, W, inputBeam_num, kernel)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N G
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=99,
            slice_num=64,
            D=51,
            H=6,
            W=6,
            inputBeam_num=1836,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_3D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W, D=D, inputBeam_num=inputBeam_num)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx
        

class Preprocess(nn.Module):
    def __init__(self, space_dim, beam_dim, elas_dim, n_hidden, act, n_layers=0):
        super().__init__()
        self.beam_process = MLP(space_dim + beam_dim, n_hidden * 2, n_hidden, n_layers=n_layers, act=act)
        self.elas_process = MLP(space_dim + elas_dim, n_hidden * 2, n_hidden, n_layers=n_layers, act=act)

    def forward(self, x_beam, x_elas):
        x_beam = self.beam_process(x_beam)
        x_elas = self.elas_process(x_elas)
        x = torch.cat([x_beam, x_elas], dim=1)
        return x


class Model(nn.Module):
    def __init__(self,
                 space_dim=3,
                 n_layers=8,
                 n_hidden=128,
                 dropout=0.0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun1_dim=3,
                 fun2_dim=6,
                 out_dim=99,
                 slice_num=64,
                 D=51,
                 H=6,
                 W=6,
                 inputBeam_num=1836,
                 ):
        super(Model, self).__init__()
        self.__name__ = 'Transolver_3D'
        self.use_checkpoint = False

        self.preprocess = Preprocess(space_dim=space_dim, beam_dim=fun1_dim, elas_dim=fun2_dim, n_hidden=n_hidden, act=act)

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      D=D,
                                                      H=H,
                                                      W=W,
                                                      inputBeam_num=inputBeam_num,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x_beam, x_elas):
        
        fx = self.preprocess(x_beam, x_elas)

        for block in self.blocks:
            if self.use_checkpoint:
                fx = checkpoint.checkpoint(block, fx)
            else:
                fx = block(fx)

        return fx
