import numpy as np
from .engine import Function

class EmbeddingLookup(Function):
    @staticmethod
    def forward(ctx, weight, indices):
        """
        weight: (num_embeddings, embedding_dim)
        indices: int tensor of shape (...,)
        returns: (..., embedding_dim)
        """
        idx = np.asarray(indices, dtype=np.int64)
        out = weight[idx]  # numpy advanced indexing
        ctx.save_for_backward(idx)
        ctx.meta["w_shape"] = weight.shape
        return out

    def backward(self, grad_out):
        (idx,) = self.ctx.saved_tensors
        num_embeddings, embed_dim = self.ctx.meta["w_shape"]
        grad_w = np.zeros((num_embeddings, embed_dim), dtype=grad_out.dtype)

        # flatten batch dims, scatter-add into grad_w
        grad_flat = grad_out.reshape(-1, embed_dim)
        idx_flat = idx.reshape(-1)
        np.add.at(grad_w, idx_flat, grad_flat)

        # No gradient for indices
        return grad_w, None

def embedding(weight, indices):
    return EmbeddingLookup.apply(weight, indices)


# class FTTransformer(nn.Module):
#     def __init__(self, cat_cardinalities: List[int], num_features: int, dim: int = 64, depth: int = 2, heads: int = 2, ff_mult: int = 2 ,dropout: float = 0.1):
#         super().__init__()
#         self.num_cat = len(cat_cardinalities)
#         self.num_num = num_features 
#         self.dim = dim

#         self.cat_embeddings = nn.ModuleList([nn.Embedding(cat_cardinality, dim) for cat_cardinality in cat_cardinalities])
#         self.num_linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_features)])

#         self.col_embed = nn.Embedding(self.num_cat + self.num_num + 1, dim) # +1 for CLS
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim)*0.02)

#         enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff_mult*dim, dropout=dropout,
#                                                batch_first=True, activation='gelu' , norm_first = True)
        
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
#         self.norm = nn.LayerNorm(dim)
#         self.drop = nn.Dropout(dropout)

#         self.cat_heads = nn.ModuleList([nn.Linear(dim, c) for c in cat_cardinalities])
#         self.num_head = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_features)])

#     def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> Tensor:
#         B = cats.size(0)
#         tokens = []

#         cls = self.cls_token.expand(B, -1, -1) # (B, 1, dim)
#         tokens.append(cls + self.col_embed(torch.zeros(B, 1, dtype=torch.long, device=cls.device))) # CLS token


#         if self.num_cat > 0:
#             cat_tok = []
#             for i in range(self.num_cat):
#                 max_index = self.cat_embeddings[i].num_embeddings - 1
#                 cats[:, i] = torch.clamp(cats[:, i], 0, max_index)
#                 emb = self.cat_embeddings[i](cats[:, i]) # (B, dim)
#                 col_id = i + 1
#                 emb = emb + self.col_embed(torch.full((B,1), col_id, dtype=torch.long, device=cats.device)).squeeze(1)
#                 cat_tok.append(emb)
#             cat_tok = torch.stack(cat_tok, dim=1)
#             tokens.append(cat_tok)

#         if self.num_num > 0:
#             num_tok = []
#             for j in range(self.num_num):
#                 x = nums[:,j:j+1]
#                 emb = self.num_linears[j](x)
#                 col_id = self.num_cat + j + 1
#                 emb = emb + self.col_embed(torch.full((B,1), col_id, dtype=torch.long, device=nums.device)).squeeze(1)
#                 num_tok.append(emb)
#             num_tok = torch.stack(num_tok, dim=1)
#             tokens.append(num_tok)

#         x = torch.cat(tokens, dim=1) # (B, num_cat + num_num + 1, dim)
#         x = self.drop(x)
#         x = self.encoder(x) # (B, num_cat + num_num + 1, dim)
#         x = self.norm(x)

#         cat_logits, num_pred = [] , []
#         for i in range(self.num_cat):
#             pos = i + 1
#             cat_logits.append(self.cat_heads[i](x[:, pos, :])) # (B, num_classes)
#         for j in range(self.num_num):
#             pos = self.num_cat + j + 1
#             num_pred.append(self.num_head[j](x[:, pos, :]).squeeze(-1)) # (B, 1)
#         cls_out = x[:, 0, :] # (B, dim)
#         return cls_out , cat_logits , num_pred
    

#     class MaskingCollator:
#         def __init__(self, mask_prob_cats: float , mask_prob_nums: float):
#             self.mpc = mask_prob_cats
#             self.mpn = mask_prob_nums
#         def __call__(self, batch):
#             cats = torch.from_numpy(np.stack([item[0] for item in batch], axis=0)).long()
#             nums = torch.from_numpy(np.stack([item[1] for item in batch], axis=0)).float()

#             B, Cc = cats.shape ; _ , n = nums.shape

#             cats_mask = torch.rand(B, Cc) < self.mpc
#             cats_mask = cats.clone()
#             cats_mask[cats_mask] = 0

#             nums_mask = torch.rand(B, n) < self.mpn
#             nums_mask = nums.clone()
#             nums_mask[nums_mask] = 0

#             targets = {
#                 "cat_targets" : cats,
#                 "num_targets" : nums,
#                 "cat_masks" : cats_mask,
#                 "num_masks" : nums_mask 
#             }
#             return (cats_mask, nums_mask), targets
