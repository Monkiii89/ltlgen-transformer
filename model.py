from torch import nn
import torch
import math
from torch import Tensor, Optional
# nn.Transformer
class Transformerdecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        # multihead_attn在后续未被调用                                       
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.functional.relu
        # 只有 norm1 和 norm3，缺少对应交叉注意力的 norm2
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm3(x + self._ff_block(x))
    # norm_first两种不同的处理方式的区别？Post-LN & Pre-LN
    # Pre-LN：先归一化再进入子层（更稳定，适合深层网络）。
    # Post-LN：先计算子层再归一化（原始 Transformer 设计）。
        return x  

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model=512, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        ####################################################### 这里的维度print一下看看
        return self.dropout(x)

class RNN(nn.Module):
    def __init__(self, d_model=512, n_head=8, embedding_size = 512, token_types=100, max_length=100, device=None):
        super(RNN, self).__init__()
        # self.transformerdecoder = nn.Sequential(
        #     Transformerdecoder(d_model=d_model, nhead=n_head, batch_first=True, device=device),
        #     Transformerdecoder(d_model=d_model, nhead=n_head, batch_first=True, device=device),
        #     Transformerdecoder(d_model=d_model, nhead=n_head, batch_first=True, device=device)
        # )
        self.transformerdecoder = Transformerdecoder(d_model=d_model, nhead=n_head, batch_first=True, device=device)
        self.out = nn.Sequential(
            nn.Linear(embedding_size, token_types),
        )
        self.embedding = nn.Embedding(num_embeddings=token_types, embedding_dim=embedding_size)
        self.position_embedding = PositionalEncoding(d_model=embedding_size, max_len=max_length)
    def forward(self, x, x_mask, y_pos_batch):
        x = self.embedding(x)
        x = self.position_embedding(x)
        # 还要加上位置编码
        
        r_out = self.transformerdecoder(x, tgt_key_padding_mask = x_mask)
        # aaa = torch.tensor([[i] for i in range(17)])
        # bbb = torch.tensor([[i] for i in range(9)] + [[i] for i in range(8)])
        # ppp = r_out[aaa,bbb,:]
        # r_out = r_out[torch.arange(x.size(0)).unsqueeze(-1),y_pos_batch,:]
        # r_out = r_out.reshape(r_out.size(0),-1)
        r_out = (r_out*(y_pos_batch.unsqueeze(-1).expand_as(r_out))).sum(dim=1)
        # r_out = r_out[:,0,:].reshape(1024,-1)
        out = self.out(r_out)
        # return nn.functional.softmax(out, dim=1)
        return out
        # return torch.sigmoid(out)
    def infv2(self, x, x_mask, y_pos_batch):
        x = self.embedding(x)
        x = self.position_embedding(x)
        
        r_out = self.transformerdecoder(x, tgt_key_padding_mask = x_mask)
        r_out = (r_out*(y_pos_batch.unsqueeze(-1).expand_as(r_out))).sum(dim=1)
        # r_out = r_out[:,0,:].reshape(1024,-1)
        out = self.out(r_out)
        return nn.functional.softmax(out, dim=1)
        # return out
        # return torch.sigmoid(out)

