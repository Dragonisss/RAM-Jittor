## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import jittor as jt
from jittor import nn
from jittor import Module
import numbers


##########################################################################
## Layer Norm

class PixelShuffle(Module):
    '''该操作将形状为 :math:`(..., C \\times r^2, H, W)` 的张量重新排列为形状 :math:`(..., C, H \\times r, W \\times r)`  的张量。的张量，其中 r 是放大因子。
    这个过程通常用于上采样或图像尺寸放大，通过调整元素位置来增加空间维度的分辨率。

    
    .. math::
        C_{out} = C_{in} / (\\text{upscale_factor})^2 \\\\
        H_{out} = H_{in} * \\text{upscale_factor} \\\\
        W_{out} = W_{in} * \\text{upscale_factor}
   
        
   
    参数:
        - upscale_factor (int): 上采样因子，即每个空间维度的放大因子

        
    代码示例:
        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = jt.randn(1,9,4,4)
        >>> output = pixel_shuffle(input)
        >>> output.shape
        [1, 1, 12, 12]
    '''
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def execute(self, x):
        n,c,h,w = x.shape
        r = self.upscale_factor
        assert c%(r*r)==0, f"input channel needs to be divided by upscale_factor's square in PixelShuffle"
        return x.reindex([n,int(c/r**2),h*r,w*r], [
            "i0",
            f"i1*{r*r}+i2%{r}*{r}+i3%{r}",
            f"i2/{r}",
            f"i3/{r}"
        ])

class PixelUnshuffle(Module):
    '''该操作是PixelShuffle的逆操作,将形状为 :math:`(..., C, H \\times r, W \\times r)` 的张量重新排列为形状 :math:`(..., C \\times r^2, H, W)` 的张量,其中 r 是缩小因子。
    这个过程通常用于下采样或图像尺寸缩小,通过调整元素位置来减少空间维度的分辨率。

    .. math::
        C_{out} = C_{in} * (\\text{downscale_factor})^2 \\\\
        H_{out} = H_{in} / \\text{downscale_factor} \\\\
        W_{out} = W_{in} / \\text{downscale_factor}

    参数:
        - downscale_factor (int): 下采样因子,即每个空间维度的缩小因子

    代码示例:
        >>> pixel_unshuffle = nn.PixelUnshuffle(3)
        >>> input = jt.randn(1,1,12,12)
        >>> output = pixel_unshuffle(input)
        >>> output.shape
        [1, 9, 4, 4]
    '''
    def __init__(self, downscale_factor):
        self.downscale_factor = downscale_factor

    def execute(self, x):
        n,c,h,w = x.shape
        r = self.downscale_factor
        assert h%r==0 and w%r==0, f"input height and width need to be divided by downscale_factor in PixelUnshuffle"
        
        # 重塑张量以准备重排
        x = x.reshape(n, c, h//r, r, w//r, r)
        # 调整维度顺序
        x = x.permute(0, 1, 3, 5, 2, 4)
        # 合并通道维度
        x = x.reshape(n, c*r*r, h//r, w//r)
        
        return x



def to_3d(x):
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h*w, c)

def to_4d(x,h,w):
    b, hw, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)

class BiasFree_LayerNorm(Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = jt.array(jt.ones(normalized_shape))

        self.normalized_shape = normalized_shape

    def execute(self, x):
        sigma = x.var(-1, keepdims=True)
        return x / jt.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = jt.array(jt.ones(normalized_shape))
        self.bias = jt.array(jt.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def execute(self, x):
        mu = x.mean(-1, keepdims=True)
        sigma = x.var(-1, keepdims=True)
        return (x - mu) / jt.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def execute(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(Module):
    def __init__(self, dim, ffn_expansion_factor, bias,finetune_type=None):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv(dim, hidden_features*2, 1, bias=bias)

        self.dwconv = nn.Conv(hidden_features*2, hidden_features*2, 3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv(hidden_features, dim, 1, bias=bias)
                
    def execute(self, x):
        x = self.project_in(x)
        x1, x2 = jt.chunk(self.dwconv(x), 2, dim=1)
        x = nn.gelu(x1) * x2
        x = self.project_out(x)
        
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = jt.array(jt.ones((num_heads, 1, 1)))

        self.qkv = nn.Conv(dim, dim*3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv(dim*3, dim*3, 3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv(dim, dim, 1, bias=bias)
        

    def execute(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = jt.chunk(qkv, 3, dim=1)   
        
        # Reshape q from [b, (head*c), h, w] to [b, head, c, (h*w)]
        q = q.reshape(b, self.num_heads, -1, q.shape[-2]*q.shape[-1])
        k = k.reshape(b, self.num_heads, -1, k.shape[-2]*k.shape[-1])
        v = v.reshape(b, self.num_heads, -1, v.shape[-2]*v.shape[-1])
        
        
        q = q.normalize(dim=-1)
        k = k.normalize(dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.softmax(attn, dim=-1)

        out = (attn @ v)
        
        # Reshape out from [b, head, c, (h*w)] to [b, (head*c), h, w]
        out = out.reshape(b, -1, h, w)

        out = self.project_out(out)
        return out



class resblock(Module):
    def __init__(self, dim):

        super(resblock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv(dim, dim, 3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv(dim, dim, 3, stride=1, padding=1, bias=False)
        )

    def execute(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv(n_feat, n_feat//2, 3, stride=1, padding=1, bias=False),
            PixelUnshuffle(2)
        )

    def execute(self, x):
        return self.body(x)

class Upsample(Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv(n_feat, n_feat*2, 3, stride=1, padding=1, bias=False),
            PixelShuffle(2)
        )

    def execute(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,finetune_type=None):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias,finetune_type)

    def execute(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv(in_c, embed_dim, 3, stride=1, padding=1, bias=bias)

    def execute(self, x):
        x = self.proj(x)

        return x




##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = jt.array(jt.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv(prompt_dim,prompt_dim,3,stride=1,padding=1,bias=False)
        

    def execute(self,x):
        B,C,H,W = x.shape

        x = x.reshape(B,C,H*W)
        emb = x.mean(dim=-1)
        prompt_weights = nn.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = jt.sum(prompt,dim=1)
        prompt = nn.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt





##########################################################################
##---------- PromptIR -----------------------
class PromptIR(Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = False,
        finetune_type = None,
        img_size = 128
    ):

        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.mask_token = jt.zeros((1, 3, img_size, img_size))
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = 384)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[0]-1 else None) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[1]-1 else None) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[2]-1 else None) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[3]-1 else None) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv(int(dim*2**1)+192, int(dim*2**2), 1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=None)
        self.reduce_noise_level3 = nn.Conv(int(dim*2**2)+512,int(dim*2**2),1,bias=bias)


        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[2]-1 else None) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv(int(dim*2**2), int(dim*2**1), 1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=None)
        self.reduce_noise_level2 = nn.Conv(int(dim*2**1)+224,int(dim*2**2),1,bias=bias)


        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[1]-1 else None) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=None)
        self.reduce_noise_level1 = nn.Conv(int(dim*2**1)+64,int(dim*2**1),1,bias=bias)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_blocks[0]-1 else None) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,finetune_type=finetune_type if i==num_refinement_blocks-1 else None) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv(int(dim*2**1), out_channels, 3, stride=1, padding=1, bias=bias)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_size = 16
        mod_pad_h = (pad_size - h % pad_size) % pad_size
        mod_pad_w = (pad_size - w % pad_size) % pad_size
        x = nn.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def execute(self, inp_img, mask=None, mask_tokens = None):

        b,c,h,w = inp_img.shape

        # Add Mask Image Modeling
        if mask is not None:
            mask = mask.repeat_interleave(1, 1).repeat_interleave(1, 2).unsqueeze(1).contiguous() 
            if mask_tokens is None:
                mask_tokens = self.mask_token.expand(b,-1,-1,-1)

            inp_img = inp_img * (1-mask) + mask_tokens * mask
        
        inp_img = self.check_image_size(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        

        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent)

            latent = jt.concat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = jt.concat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = jt.concat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = jt.concat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
           
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = jt.concat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = jt.concat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1[:,:,:h,:w]
