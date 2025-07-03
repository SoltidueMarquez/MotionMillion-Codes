import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from models.encdec import Encoder


class Text2Motion_Transformer_root(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature, root_motion, text_cond_prob, root_cond_prob, force_mask_text, force_mask_root):
        feat = self.trans_base(idxs, clip_feature, root_motion, text_cond_prob, root_cond_prob, force_mask_text, force_mask_root)
        logits = self.trans_head(feat)
        return logits

    def sample(self, clip_feature, root_motion, root_cond_prob, text_cond_prob, force_mask_text, force_mask_root, if_categorial=False):
        
        for k in range(self.block_size):
            if k == 0 or k == 1:
                x = []
            else:
                x = xs

            logits = self.forward(x, clip_feature, root_motion, text_cond_prob, root_cond_prob, force_mask_text, force_mask_root)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
                
            # append to the sequence and continue
            if k == 1 or k == 0:
                xs = idx
            elif k % 2 == 1:
                xs = torch.cat((xs, idx), dim=1)
            
            # if k == self.block_size - 2:
            #     return xs[:, :-1]
            if k == self.block_size - 4:
                return xs
            # assert root_motion.shape[1] % 4 == 0
            # if k > 1:
            #     if len(xs) == root_motion.shape[1] / 4:
            #         return xs
        return xs
    
    def sample_inference(self, clip_feature, root_motion, root_cond_prob, text_cond_prob, force_mask_text, force_mask_root, if_categorial=False):
        
        for k in range(self.block_size):
            if k == 0 or k == 1:
                x = []
            else:
                x = xs
            logits = self.forward(x, clip_feature, root_motion, text_cond_prob, root_cond_prob, force_mask_text, force_mask_root)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue

            if k == 1:
                xs = idx
            elif k % 2 == 1:
                xs = torch.cat((xs, idx), dim=1)
            
            # if k == self.block_size - 2:
            #     return xs[:, :-1]
            # if k == self.block_size - 4:
            #     return xs
            assert root_motion.shape[1] % 4 == 0
            
            if k > 1:
                if xs.shape[1] == root_motion.shape[1] / 4:
                    return xs
        return xs

    def sample_inference_cond(self, clip_feature, root_motion, text_cond_scale, root_cond_scale, if_categorial=False):
        
        for k in range(self.block_size):
            if k == 0 or k == 1:
                x = []
            else:
                x = xs

            logits = self.forward(x, clip_feature, root_motion, text_cond_prob=0, root_cond_prob=0, force_mask_text=False, force_mask_root=False) # with root and text
            text_logits = self.forward(x, clip_feature, root_motion, text_cond_prob=0, root_cond_prob=0, force_mask_root=True, force_mask_text=False) # only with text 
            root_logits = self.forward(x, clip_feature, root_motion, text_cond_prob=0, root_cond_prob=0, force_mask_root=False, force_mask_text=True) # only with root 
            neither_logits = self.forward(x, clip_feature, root_motion, text_cond_prob=0, root_cond_prob=0, force_mask_root=True, force_mask_text=True) # no root and text
            
            
            logits = text_cond_scale * text_logits + (root_cond_scale - text_cond_scale) * root_logits 
            
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=2, dim=-1)
                
                if idx[0, 0] == self.num_vq:
                    idx= idx[0, 1].unsqueeze(0).unsqueeze(0)
                else:
                    idx = idx[0, 0].unsqueeze(0).unsqueeze(0)
            # append to the sequence and continue

            if k == 1:
                xs = idx
            elif k % 2 == 1:
                xs = torch.cat((xs, idx), dim=1)
            
            # if k == self.block_size - 2:
            #     return xs[:, :-1]
            # if k == self.block_size - 4:
            #     return xs
            assert root_motion.shape[1] % 4 == 0
            
            if k > 1:
                if xs.shape[1] == root_motion.shape[1] / 4:
                    return xs
        return xs

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class DoubleCausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask_register = torch.tril(torch.ones(block_size, block_size))

        for i in range(block_size-1):
            if i % 2 == 1:
                mask_register[i, i+1] = 1

        self.register_buffer("mask", mask_register.view(1, 1, block_size, block_size))

        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class root_encoder(nn.Module):
    
    def __init__(self, down_t=4, embed_dim=512):
        super().__init__()
        self.root_emb = nn.Linear(4, embed_dim)

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)


    def forward(self, root_motion):
        root_motion = self.root_emb(root_motion)
        return root_motion

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)

        self.root_encoder = Encoder(4, output_emb_width=embed_dim, down_t=2, stride_t=2, width=embed_dim, depth=1, dilation_growth_rate=3, activation='relu')

        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _mask_cond_root(self, cond, cond_drop_prob, force_mask_root):
        bs, _, _ =  cond.shape
        if force_mask_root:
            return torch.ones_like(cond) * 100
        elif cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_drop_prob).view(bs, 1, 1)
            return cond * (1. - mask) + torch.ones_like(cond) * 100 * mask
        else:
            return cond
        
    def _mask_cond_text(self, cond, cond_drop_prob, force_mask_text=False):
        bs, _ =  cond.shape
        if force_mask_text:
            return torch.zeros_like(cond)
        elif cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, idx, clip_feature, root_motion, text_cond_prob, root_cond_prob, force_mask_text, force_mask_root):
        
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert 2*t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx) # 128, 50, 1024
            root_embeddings = self.root_encoder(root_motion.permute(0, 2, 1)).permute(0,2,1)[:, :token_embeddings.shape[1], :]# b, 50, 1024

            # mask root
            root_embeddings = self._mask_cond_root(root_embeddings, root_cond_prob, force_mask_root)

            text_cond = self.cond_emb(clip_feature)

            # mask text
            text_cond = self._mask_cond_text(text_cond, text_cond_prob, force_mask_text)

            # interleave token embeddings with the root embeddings
            stacked = torch.stack((token_embeddings, root_embeddings), dim=2)

            interleaved = stacked.view(b, -1, root_embeddings.shape[-1])

            # token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), interleaved], dim=1)
            
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    


        

