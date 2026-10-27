import math
import torch
import numpy as np
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=None, rel_pos_spatial=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial = rel_pos_spatial
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.window_size = window_size
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'B N (K H D) -> K B H N D', K=3, H=self.num_heads)
        q, k, v = qkv.unbind(0)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, rel_pos_spatial=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial=rel_pos_spatial

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B_, N, C = x.shape
        x_reshaped = x.reshape(B_, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x_padded = F.pad(x_reshaped, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x_padded.shape

        x_windows = window_partition(x_padded, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        B_w = x_windows.shape[0]
        N_w = x_windows.shape[1]

        qkv = self.qkv(x_windows).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)

        x_attn = self.proj(x_attn)

        x_attn = x_attn.view(-1, self.window_size[0], self.window_size[1], C)
        x_rev = window_reverse(x_attn, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x_rev = x_rev[:, :H, :W, :].contiguous()

        x_final = x_rev.view(B_, H * W, C)

        return x_final

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, window=False, rel_pos_spatial=False):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial)
        else:
            self.attn = WindowAttention(
                dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                rel_pos_spatial=rel_pos_spatial,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, H, W, mask=None):

        att_out = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(att_out)

        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)

        x_conv_in = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()


        x_conv_out = self.conv_layer(x_conv_in)

        x_conv_out = rearrange(x_conv_out, 'b c h w -> b (h w) c')

        x = x + x_conv_out

        return x


class HighFreqEnhance(nn.Module):
    def __init__(self, hf_strength=0.5, radius_threshold=0.1):
        """
        hf_strength: 高频增强强度权重（添加多少高频成分）
        radius_threshold: 高频阈值，低于该频率半径的成分被认为是低频
        """
        super().__init__()
        self.hf_strength = hf_strength
        self.radius_threshold = radius_threshold

    def forward(self, x):
        B, C, H, W = x.shape

        y = torch.linspace(-0.5, 0.5, steps=H, device=x.device)
        x_ = torch.linspace(-0.5, 0.5, steps=W, device=x.device)
        Y, X = torch.meshgrid(y, x_)
        radius = torch.sqrt(X**2 + Y**2)

        hf_mask = (radius > self.radius_threshold).float()
        hf_mask = hf_mask.unsqueeze(0).unsqueeze(0)

        x_fft = torch.fft.fft2(x, norm='ortho')

        x_fft_hf = x_fft * hf_mask

        x_hf = torch.fft.ifft2(x_fft_hf, norm='ortho').real

        out = x + self.hf_strength * x_hf
        return out




def aggregate_temporal_tokens(x_temporal_attn, batch_size, num_spatial_tokens, mode):
    x_temporal = rearrange(
        x_temporal_attn, '(b l) t c -> b l t c', b=batch_size, l=num_spatial_tokens
    )
    if mode == 'last':
        return x_temporal[:, :, -1, :]
    if mode == 'mean':
        return x_temporal.mean(dim=2)
    raise ValueError(f"Unsupported temporal_pool mode: {mode}")

def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h, grid_w = grid_size


    grid_y = np.arange(grid_h, dtype=np.float32)
    grid_x = np.arange(grid_w, dtype=np.float32)


    assert embed_dim % 2 == 0, 'Embedding dimension must be even for sin-cos encoding'
    half_dim = embed_dim // 2

    div_term = np.exp(np.arange(0, half_dim, 2) * -(math.log(10000.0) / half_dim))

    pe_x = np.sin(grid_x[:, np.newaxis] * div_term)
    pe_x_cos = np.cos(grid_x[:, np.newaxis] * div_term)
    pe_x = np.concatenate([pe_x, pe_x_cos], axis=-1)

    pe_y = np.sin(grid_y[:, np.newaxis] * div_term)
    pe_y_cos = np.cos(grid_y[:, np.newaxis] * div_term)
    pe_y = np.concatenate([pe_y, pe_y_cos], axis=-1)

    pos_embed = np.concatenate([pe_y[:, None, :].repeat(grid_w, axis=1), pe_x[None, :, :].repeat(grid_h, axis=0)], axis=-1)

    pos_embed = pos_embed.reshape(-1, embed_dim)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class BandSelectionBranch(nn.Module):
    def __init__(self, in_bands, use_sigmoid=True, init_scale=1.0):
        super().__init__()
        self.in_bands = in_bands
        self.use_sigmoid = use_sigmoid
        self.logits = nn.Parameter(torch.zeros(in_bands))
        self.init_scale = init_scale

    def forward(self, x):
        bw = torch.sigmoid(self.logits) if self.use_sigmoid else self.logits
        bw = bw * self.init_scale
        return x * bw.unsqueeze(0), bw


class SpectralEncoder(nn.Module):
    def __init__(self, in_bands, embed_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_bands, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class STAR(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_spectral_channels = model_config['in_channels']
        self.drop_last_input_channel = model_config.get('drop_last_input_channel', True)
        self.num_time_steps = model_config['max_seq_len']
        self.img_size = to_2tuple(model_config['img_res'])
        self.patch_size = to_2tuple(model_config['patch_size'])
        self.embed_dim = model_config['dim']
        self.num_classes = model_config['num_classes']
        self.mlp_ratio = model_config.get('mlp_ratio', 4.)
        self.drop_path_rate = model_config.get('drop_path_rate', 0.1)
        self.temporal_attn_heads = model_config.get('temporal_attn_heads', 8)
        self.temporal_attn_layers = model_config.get('temporal_attn_layers', 3)
        self.spatial_encoder_layers = model_config.get('spatial_encoder_layers', 8)
        self.spatial_decoder_layers = model_config.get('spatial_decoder_layers', 4)
        self.spatial_attn_heads = model_config.get('spatial_attn_heads', 12)
        self.cnn_stem_channels = model_config.get('cnn_stem_channels', [64, 128])
        self.use_abs_pos_emb = model_config.get('use_abs_pos_emb', True)
        self.learnable_pos = model_config.get('learnable_pos', False)
        self.window_attention = model_config.get('window_attention', True)
        self.window_size = to_2tuple(model_config.get('window_size', (self.img_size[0] // 16, self.img_size[1] // 16)))
        self.interval = model_config.get('interval', 1)
        self.hf_strength = model_config.get('hf_strength', 0.5)
        self.hf_radius_threshold = model_config.get(
            'hf_radius_threshold', model_config.get('radius_threshold', 0.1)
        )
        self.temporal_pool = model_config.get('temporal_pool', 'mean')
        if self.temporal_pool not in {'last', 'mean'}:
            raise ValueError(
                f"Unsupported temporal_pool '{self.temporal_pool}'. Expected one of: last, mean."
            )

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.stem_downsample_factor = 2
        self.stem_out_H = self.img_size[0] // self.stem_downsample_factor
        self.stem_out_W = self.img_size[1] // self.stem_downsample_factor
        self.num_spatial_patches = self.stem_out_H * self.stem_out_W

        in_c = self.num_spectral_channels

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_c, self.cnn_stem_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.cnn_stem_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.cnn_stem_channels[0], self.cnn_stem_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.cnn_stem_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.high_freq_enhance = HighFreqEnhance(
            hf_strength=self.hf_strength,
            radius_threshold=self.hf_radius_threshold,
        )
        self.stem_out_channels = self.cnn_stem_channels[-1]

        # Retained from training for checkpoint and parameter-count compatibility.
        self.num_bands = self.num_spectral_channels
        self.band_selector = BandSelectionBranch(self.num_bands, use_sigmoid=True, init_scale=1.0)
        self.spectral_embed_dim = min(self.embed_dim, 128)
        self.spectral_encoder = SpectralEncoder(self.num_bands, self.spectral_embed_dim, hidden=128)
        self.spectral_decoder = nn.Sequential(
            nn.Linear(self.spectral_embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.num_bands),
        )
        self.spectral_to_embed = nn.Linear(self.spectral_embed_dim, self.embed_dim)

        self.temporal_patch_embed = nn.Linear(self.stem_out_channels, self.embed_dim)

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_time_steps, self.embed_dim))
        trunc_normal_(self.temporal_pos_embed, std=.02)

        dpr_temporal = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.temporal_attn_layers)]
        self.temporal_blocks = nn.ModuleList([
            Block(
             dim=self.embed_dim, num_heads=self.temporal_attn_heads, mlp_ratio=self.mlp_ratio, qkv_bias=True,
             drop_path=dpr_temporal[i], norm_layer=norm_layer,
             window_size=(1, self.num_time_steps), window=False,
             act_layer=nn.GELU
            ) for i in range(self.temporal_attn_layers)
        ])

        if self.use_abs_pos_emb:
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial_patches, self.embed_dim), requires_grad=self.learnable_pos)
            pos_embed_np = _get_2d_sincos_pos_embed(self.embed_dim, (self.stem_out_H, self.stem_out_W), cls_token=False)
            self.spatial_pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float().unsqueeze(0))
        else:
            self.spatial_pos_embed = None

        dpr_spatial_enc = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.spatial_encoder_layers)]
        self.spatial_encoder_blocks = nn.ModuleList([
            Block(dim=self.embed_dim, num_heads=self.spatial_attn_heads, mlp_ratio=self.mlp_ratio, qkv_bias=True,
                  drop_path=dpr_spatial_enc[i], norm_layer=norm_layer,
                  window_size=self.window_size if self.window_attention else (self.stem_out_H, self.stem_out_W),
                  window=self.window_attention,
                  act_layer=nn.GELU) for i in range(self.spatial_encoder_layers)
        ])

        dpr_spatial_dec = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.spatial_decoder_layers)]
        self.spatial_decoder_blocks = nn.ModuleList([
            Block(dim=self.embed_dim, num_heads=self.spatial_attn_heads, mlp_ratio=self.mlp_ratio, qkv_bias=True,
                  drop_path=dpr_spatial_dec[i], norm_layer=norm_layer,
                  window_size=self.window_size if self.window_attention else (self.stem_out_H, self.stem_out_W),
                  window=self.window_attention,
                  act_layer=nn.GELU) for i in range(self.spatial_decoder_layers)
        ])

        self.upsample_conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(),
        )
        self.upsample_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 4),
            nn.ReLU(),
        )
        self.upsample_conv3 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim // 4, self.embed_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.embed_dim // 8),
            nn.ReLU(),
        )
        self.classification_head = nn.Conv2d(self.embed_dim // 2, self.num_classes, kernel_size=1)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.temporal_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        for layer_id, layer in enumerate(self.spatial_encoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        for layer_id, layer in enumerate(self.spatial_decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):


        x = x.permute(0, 1, 4, 2, 3)
        if self.drop_last_input_channel:
            x_t = x[:, :, :-1, :, :]
        else:
            x_t = x

        B, T, C, H, W = x_t.shape

        x_flat_bt = rearrange(x_t, 'b t c h w -> (b t) c h w')

        feat_stage2 = self.cnn_stem(x_flat_bt)


        cnn_features = self.high_freq_enhance(feat_stage2)
        _, C_stem, H_s1, W_s1 = cnn_features.shape

        x_temporal_in = rearrange(cnn_features, 'bt c h w -> bt (h w) c')

        x_temporal_in = self.temporal_patch_embed(x_temporal_in)

        x_temporal_attn = rearrange(x_temporal_in, '(b t) l c -> (b l) t c', b=B, t=T)

        x_temporal_attn = x_temporal_attn + self.temporal_pos_embed

        for blk in self.temporal_blocks:
            x_temporal_attn = blk(x_temporal_attn, H=1, W=T)

        x_spatial_in = aggregate_temporal_tokens(
            x_temporal_attn,
            batch_size=B,
            num_spatial_tokens=H_s1 * W_s1,
            mode=self.temporal_pool,
        )

        if self.spatial_pos_embed is not None:

            x_spatial_enc = x_spatial_in + self.spatial_pos_embed
        else:
            x_spatial_enc = x_spatial_in

        for blk in self.spatial_encoder_blocks:
            x_spatial_enc = blk(x_spatial_enc, H_s1, W_s1)

        x_spatial_dec = x_spatial_enc
        for blk in self.spatial_decoder_blocks:
            x_spatial_dec = blk(x_spatial_dec, H_s1, W_s1)

        x_upsample_in = rearrange(x_spatial_dec, 'b (h w) c -> b c h w', h=H_s1, w=W_s1)



        x_up1 = self.upsample_conv1(x_upsample_in)

        output = self.classification_head(x_up1)



        return output
