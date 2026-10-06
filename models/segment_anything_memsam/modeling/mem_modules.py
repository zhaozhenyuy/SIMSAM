import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet
from einops import rearrange


class BasicConv(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class APFE(nn.Module):
    """Anatomical Prior-aware Feature Enhancement from OSA.

    It decomposes a feature map into positive and negative residuals around a
    local acoustic field, then fuses the two branches with a pixel-wise gate.
    """

    def __init__(self, channels, kernel_size=15):
        super().__init__()
        padding = kernel_size // 2
        self.local_field = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        self.pos_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.neg_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        acoustic_field = self.local_field(x)
        x_pos = F.relu(x - acoustic_field)
        x_neg = F.relu(acoustic_field - x)
        h_pos = self.pos_branch(x_pos)
        h_neg = self.neg_branch(x_neg)
        weight = self.gate(torch.cat([h_pos, h_neg], dim=1))
        return weight * h_pos + (1.0 - weight) * h_neg


class OrthogonalizedPromptStabilizer(nn.Module):
    """OSU-style projection for the SAM dense memory prompt.

    This keeps the memory prompt close to the Stiefel manifold along channel
    directions using Frobenius normalization and Newton-Schulz iterations.
    """

    def __init__(
        self,
        channels,
        num_iters=5,
        residual_init=0.1,
        eps=1e-6,
        ns_type="classic",
        safety_margin=0.95,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
        self.ns_type = ns_type
        self.safety_margin = safety_margin
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init)))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def _orthogonalize(self, matrix):
        # matrix: B x rows x cols, rows >= cols is preferred.
        orig_dtype = matrix.dtype
        matrix = matrix.float()
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = self.safety_margin * matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + self.eps)
        transposed = matrix.shape[-2] < matrix.shape[-1]
        if transposed:
            matrix = matrix.transpose(-2, -1)

        identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        identity = identity.unsqueeze(0).expand(matrix.shape[0], -1, -1)
        x = matrix
        target_norm = math.sqrt(float(matrix.shape[-1]))
        max_norm = 1.05 * target_norm
        for _ in range(self.num_iters):
            xtx = x.transpose(-2, -1).matmul(x)
            if self.ns_type == "quintic":
                a, b, c = 3.4445, -4.7750, 2.0315
                x = a * x + b * x.matmul(xtx) + c * x.matmul(xtx.matmul(xtx))
            else:
                x = 0.5 * x.matmul(3.0 * identity - xtx)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            current_norm = x.norm(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            x = x * (max_norm / current_norm).clamp(max=1.0)

        if transposed:
            x = x.transpose(-2, -1)
        return x.to(orig_dtype)

    def forward(self, x):
        if x.dim() == 5:
            batch_size, num_objects = x.shape[:2]
            x_flat = x.flatten(start_dim=0, end_dim=1)
            x_flat = self.forward(x_flat)
            return x_flat.view(batch_size, num_objects, *x_flat.shape[1:])
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D or 5D memory prompt, got shape {tuple(x.shape)}")

        b, c, h, w = x.shape
        matrix = x.flatten(2).transpose(1, 2)  # B x HW x C
        ortho = self._orthogonalize(matrix)
        ortho = ortho.transpose(1, 2).view(b, c, h, w)
        gate = 0.2 * torch.sigmoid(self.residual_gate)
        scale = self.scale.clamp(0.25, 4.0)
        return x + gate * (ortho * scale)


class OSUMemoryState(nn.Module):
    """Continuous OSU memory state used as a replacement for memory-bank readout.

    The state has shape B x N x Cv x Ck. Each update first performs the
    Euclidean linear recurrent step from key/value tokens, then projects the
    state back to a scaled Stiefel manifold with Newton-Schulz iterations.
    """

    def __init__(
        self,
        value_dim,
        key_dim,
        num_iters=5,
        alpha=0.98,
        beta=1.0,
        state_scale=2.0,
        eps=1e-6,
        ns_type="classic",
        safety_margin=0.95,
    ):
        super().__init__()
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.num_iters = num_iters
        self.eps = eps
        self.ns_type = ns_type
        self.safety_margin = safety_margin
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.state_scale = nn.Parameter(torch.tensor(float(state_scale)))

    def init_state(self, batch_size, num_objects, device, dtype):
        return torch.zeros(
            batch_size,
            num_objects,
            self.value_dim,
            self.key_dim,
            device=device,
            dtype=dtype,
        )

    def _orthogonalize(self, state):
        batch_size, num_objects, value_dim, key_dim = state.shape
        matrix = state.flatten(start_dim=0, end_dim=1)
        orig_dtype = matrix.dtype
        matrix = matrix.float()
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = self.safety_margin * matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + self.eps)

        identity = torch.eye(key_dim, device=matrix.device, dtype=matrix.dtype)
        identity = identity.unsqueeze(0).expand(matrix.shape[0], -1, -1)
        x = matrix
        target_norm = math.sqrt(float(key_dim))
        max_norm = 1.05 * target_norm
        for _ in range(self.num_iters):
            xtx = x.transpose(-2, -1).matmul(x)
            if self.ns_type == "quintic":
                a, b, c = 3.4445, -4.7750, 2.0315
                x = a * x + b * x.matmul(xtx) + c * x.matmul(xtx.matmul(xtx))
            else:
                x = 0.5 * x.matmul(3.0 * identity - xtx)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            current_norm = x.norm(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            x = x * (max_norm / current_norm).clamp(max=1.0)

        x = x.to(orig_dtype).view(batch_size, num_objects, value_dim, key_dim)
        return self.state_scale.clamp(0.25, 4.0) * x

    def update(self, state, key, value):
        # key: B x Ck x H x W, value: B x N x Cv x H x W
        key = torch.nan_to_num(key, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        key_tokens = F.normalize(key.flatten(2), dim=1, eps=self.eps)
        value_tokens = value.flatten(3).clamp(-10.0, 10.0)
        token_count = key_tokens.shape[-1]

        key_cov = torch.bmm(
            key_tokens,
            key_tokens.transpose(1, 2),
        ) / float(token_count)
        value_key = torch.einsum(
            "bnvl,bkl->bnvk",
            value_tokens,
            key_tokens,
        ) / float(token_count)

        identity = torch.eye(
            self.key_dim,
            device=key.device,
            dtype=key.dtype,
        ).unsqueeze(0)
        alpha = self.alpha.clamp(0.0, 0.995)
        beta = self.beta.clamp(0.0, 1.0)
        transition = identity - beta * key_cov
        state_euc = alpha * torch.matmul(state, transition.unsqueeze(1)) + beta * value_key
        return self._orthogonalize(state_euc)

    def read(self, state, query_key):
        # query_key: B x Ck x H x W
        batch_size, _, height, width = query_key.shape
        query_key = torch.nan_to_num(query_key, nan=0.0, posinf=0.0, neginf=0.0)
        query_tokens = F.normalize(query_key.flatten(2), dim=1, eps=self.eps)
        memory_tokens = torch.einsum("bnvk,bkl->bnvl", state, query_tokens)
        memory_tokens = torch.nan_to_num(memory_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        return memory_tokens.view(batch_size, state.shape[1], self.value_dim, height, width)


def interpolate_groups(g, ratio, mode, align_corners):
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1), 
                scale_factor=ratio, mode=mode, align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g

def upsample_groups(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate_groups(g, ratio, mode, align_corners)

def downsample_groups(g, ratio=1/2, mode='area', align_corners=None):
    return interpolate_groups(g, ratio, mode, align_corners)


class GConv2D(nn.Conv2d):
    def forward(self, g):
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2D(out_dim, out_dim, kernel_size=3, padding=1)
 
    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        
        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class MainToGroupDistributor(nn.Module):
    def __init__(self, x_transform=None, method='cat', reverse_order=False):
        super().__init__()

        self.x_transform = x_transform
        self.method = method
        self.reverse_order = reverse_order

    def forward(self, x, g):
        num_objects = g.shape[1]

        if self.x_transform is not None:
            x = self.x_transform(x)

        if self.method == 'cat':
            if self.reverse_order:
                g = torch.cat([g, x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1)], 2)
            else:
                g = torch.cat([x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1), g], 2)
        elif self.method == 'add':
            g = x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1) + g
        else:
            raise NotImplementedError

        return g


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g) # 1,1,512,h,w
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        # g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
        #     self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g[0], h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False, enable_apfe=False, apfe_kernel_size=7):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        # self.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
        #         nn.ReLU()
        #     )

        self.distributor = MainToGroupDistributor()
        self.apfe = APFE(256, kernel_size=apfe_kernel_size) if enable_apfe else None
        self.fuser = FeatureFusionBlock(128, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g = self.layer1(g) # 1/4
        g = self.layer2(g) # 1/8
        # g = self.layer3(g) # 1/16

        g = g.view(batch_size, num_objects, *g.shape[1:])
        # image_feat = self.downsample(image_feat)
        if self.apfe is not None:
            image_feat = self.apfe(image_feat)
        g = self.fuser(image_feat, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits

class MemoryDecoder(nn.Module):
    def __init__(self, val_dim, hidden_dim, enable_osu_prompt=False, osu_iters=5, osu_ns_type="classic"):
        super().__init__()

        self.fuser = FeatureFusionBlock(256, val_dim+hidden_dim, 512, 256)
        self.prompt_stabilizer = (
            OrthogonalizedPromptStabilizer(256, num_iters=osu_iters, ns_type=osu_ns_type)
            if enable_osu_prompt
            else None
        )
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater(256, hidden_dim)
        else:
            self.hidden_update = None

    def forward(self, imge, hidden_state, memory_readout, h_out=True):
        if self.hidden_update is not None:
            g16 = self.fuser(imge, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(imge, memory_readout)

        if h_out and self.hidden_update is not None:
            hidden_state = self.hidden_update([g16], hidden_state)
        else:
            hidden_state = None

        if self.prompt_stabilizer is not None:
            g16 = self.prompt_stabilizer(g16)
        
        return hidden_state, g16
