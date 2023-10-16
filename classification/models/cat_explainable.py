""" 
Code inspired by:
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce, Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

def mean_pool(x, stride=2):
    x = rearrange(x, 'b (n d) f -> b n d f', d = stride)
    x = x.mean(dim = -2)
    return x


def dropout(data, p, b, training):
    org_shape = data.shape
    data = data.view(b, -1)
    data = F.dropout(data, p=p, training=training)
    data = data.view(org_shape)
    return data

# main class
class CATModule(nn.Module):
    def __init__(
        self,
        *,
        num_latents,
        input_channels,
        input_channels_feats,
        depth = 1,
        latent_dim = 512,
        latent_heads = 8,
        latent_dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        self_per_cross_attn = 1,
        cross_heads = 1,
        cross_dim_head = 64,
        use_skip_connection = False,
        compression_factor = 1,
        append_org_input = True,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)
        Args:
          depth: Depth of net.
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          self_per_cross_attn: Number of self attention blocks per cross attn.
        """
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.use_skip_connection = use_skip_connection
        self.compression_factor = compression_factor
        self.append_org_input = append_org_input 

        feats_input_dim = input_channels

        get_feats_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, feats_input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = feats_input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_feats_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_feats_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth): 
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_feats_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        ############## Stuff for feeding in the original input ##############

        if self.append_org_input:
            get_feats_input_cross_attn = lambda : PreNorm(latent_dim, Attention(latent_dim, input_channels_feats, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_channels_feats)
            get_feats_input_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
            get_feats_input_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
            get_feats_input_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

            get_feats_input_cross_attn, get_feats_input_cross_ff, get_feats_input_latent_attn, get_feats_input_latent_ff = map(cache_fn, (get_feats_input_cross_attn, get_feats_input_cross_ff, get_feats_input_latent_attn, get_feats_input_latent_ff))

            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_feats_input_latent_attn(**cache_args, key = block_ind),
                    get_feats_input_latent_ff(**cache_args, key = block_ind)
                ]))
        
            self.feats_input_cross_attn = get_feats_input_cross_attn(**cache_args)
            self.feats_input_cross_ff = get_feats_input_cross_ff(**cache_args)
            self.feats_input_self_attns = self_attns

        ####################################################################

        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.pos_embedding_cls = nn.Parameter(torch.randn(1, 1, latent_dim))


    def forward(
        self,
        context,
        feats_data,
        mask = None,
    ):
        b, *_, _, device, dtype = *context.shape, context.device, context.dtype

        x = repeat(self.latents, 'n d -> b n d', b = b)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        cls_tokens = cls_tokens + self.pos_embedding_cls[:, :1]

        x = torch.cat((cls_tokens, x), dim = 1)

        # layers
        for idx, (cross_attn, cross_ff, self_attns) in enumerate(self.layers):
            x = cross_attn(x, context = context, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
            

        if self.use_skip_connection: 
            compressed_context = mean_pool(context, stride = self.compression_factor)
            x[:, 1:] = x[:, 1:] + compressed_context

        # Add the original input
        if self.append_org_input:
            cross_attn, cross_ff, self_attns = self.feats_input_cross_attn, self.feats_input_cross_ff, self.feats_input_self_attns
            x = cross_attn(x, context = feats_data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
        
        return x



class CATExplainable(nn.Module):
    def __init__(
        self,
        *,
        depth,
        feats_module,
        cat_module,
        self_supervision,
        num_classes = 1000,
        fourier_encode_data = True,
        final_classifier_head = True,
        deep_supervision = False,
        use_dropout = False,
        p_do = 0.0,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)
        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_classes: Output number of classes.
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.fourier_encode_data = fourier_encode_data
        self.deep_supervision = deep_supervision
        self.use_dropout = use_dropout
        self.p_do = p_do
        self.self_supervision = self_supervision

        # Feats
        self.feats_input_axis = feats_module.input_axis
        self.feats_max_freq = feats_module.max_freq
        self.feats_num_freq_bands = feats_module.num_freq_bands

        feats_fourier_channels = (feats_module.input_axis * ((feats_module.num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        feats_input_dim = feats_fourier_channels + feats_module.input_channels # This is the input dimension to the CAT_Module

        num_latents = feats_module.num_latents
        self.layers = nn.ModuleList([])
        
        compression_factor = feats_module.compression_factor
        for i in range(depth):
            if i == 0:
                input_channels = feats_input_dim
                use_skip_connection = False
            else:
                input_channels = cat_module.latent_dim
                use_skip_connection = feats_module.use_skip_connection
            self.layers.append(
                CATModule(
                    num_latents = num_latents,
                    input_channels=input_channels,
                    use_skip_connection=use_skip_connection,
                    compression_factor=compression_factor,
                    input_channels_feats = feats_input_dim,
                    append_org_input=True,
                    **cat_module, 
                )
            )
            # Half the number of latents at each stop
            num_latents = num_latents // feats_module.compression_factor
            if num_latents <= feats_module.min_num_latents:
                print(f"Note: The chosen CAT depths leads to a small number of latents in layer {i} (Number of latents: {num_latents}") # . We thus set them to be equal to {feats_module.min_num_latents}")
                #num_latents = feats_module.min_num_latents
                # TODO: This is problematic. Maybe remove the min_num_latents
                #compression_factor = 1
        
        if self.self_supervision.use:
            self.layers.append(
                CATModule(
                    num_latents = self.self_supervision.num_features,
                    input_channels=input_channels,
                    use_skip_connection=False,
                    latent_dim=feats_module.input_channels,
                    input_channels_feats = feats_input_dim,
                    append_org_input = False,
                    **self.self_supervision.cat_module,
                )
            )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(cat_module.latent_dim),
            nn.Linear(cat_module.latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()



    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False
    ):
        feats_data, _ = data[0], data[1]

        b, *_, _, device, dtype = *feats_data.shape, feats_data.device, feats_data.dtype
        
        feats_axis = (feats_data.shape[1], )

        assert len(feats_axis) == self.feats_input_axis, 'Feats input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            # feats
            feats_axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), feats_axis))
            feats_pos = torch.stack(torch.meshgrid(*feats_axis_pos, indexing = 'ij'), dim = -1)
            feats_enc_pos = fourier_encode(feats_pos, self.feats_max_freq, self.feats_num_freq_bands)
            feats_enc_pos = rearrange(feats_enc_pos, '... n d -> ... (n d)')
            feats_enc_pos = repeat(feats_enc_pos, '... -> b ...', b = b)
            feats_data = torch.cat((feats_data, feats_enc_pos), dim = -1)

        # concat to channels of data and flatten axis
        feats_data = rearrange(feats_data, 'b ... d -> b (...) d')

        # dropout
        if self.use_dropout:
            if self.self_supervision.use:
                training = True
            else:
                training = self.training
            feats_data = dropout(data=feats_data, p=self.p_do, b=b, training=training)
  
        logits=[]
        # layers
        context = feats_data
        for idx, (cat_module) in enumerate(self.layers):
            x = cat_module (context = context, mask = mask, feats_data = feats_data)
            if not self.self_supervision.use:
                logits.append(self.to_logits(x[:, 0]))
            context = x[:, 1:]
        
        if self.self_supervision.use:
            return x[:, 1:]

        # allow for fetching embeddings
        if return_embeddings:
            return x

        # to logits
        if self.deep_supervision:
            return logits
        else:
            return logits[-1]
        #return self.to_logits(x)