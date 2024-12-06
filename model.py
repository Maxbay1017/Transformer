from turtle import forward
import torch
import torch.nn as nn
import math

# 1. 针对token idx的embedding层 -> [12,33,252,...,23] (batch_size , seq_len) -> (batch_size, seq_len, d_model)
# 1.1. build a (vocab_size, d_model) d_model为词嵌入的维度 通过look-up 输出 token_idx的词嵌入的Matrix
class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

# 2. 位置编码, 使model感知到句子中每个token的相对位置与绝对位置 基于Sin(X)实现 最终与inputembedding相加
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model 
        self.seq_len = seq_len 
        self.drop_out = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model) 初始化一个PE表用于存储positionalencoding matrix
        pe = torch.zeros(seq_len, d_model) 
        # create a vector of shape (seq_len)
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # (seq_len,1)   pos 表示每个token在句子中的位置
        # create a diver item 用于建立分母 与 pos相乘
        div_term = torch.exp( torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model) )
        # apply sine to even indices
        pe[:,0::2] = torch.sin(position * div_term)
        # apply cosine to odd indices
        pe[:,1::2] = torch.cos(position * div_term)
        # Add a batch dimesion to the positional ecoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe',pe)

    def forward(self, x): # the x is input embedding  (batch_size, seq_len, d_model)
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad(False)
        return self.drop_out(x)

# 3. 多头注意力机制, 包含padding_mask与casual_mask
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        # make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # dimeison of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod # 静态方法使用 @staticmethod 装饰器进行定义，不需要访问类的实例属性或方法。静态方法可以直接通过类名调用，而不需要创建类的实例。
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # (batch, h, seq_len, d_k)
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q,k,v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #(batch,seq_len,d_model) -> (batch,seq_len,h,d_k)->(batch,h,seq_len,d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        # calcuate the attention score
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # combine all the heads together
        # (batch,h,seq_len, d_k) -> (batch,seq_len,h,d_k) -> (batch,seq_len,d_model) contiguous 是 PyTorch 中的一个方法，用于确保张量在内存中是连续的。 view 是 PyTorch 中的一个方法，用于重塑张量的形状。
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch,seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
# 4. 层标准化
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# 5. 残差链接
class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # pre_norm is easier for traning!

# 6. FFN层
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

# 7. EncoderBlock 构建Encoder编码器的最小组成单元unit
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])    

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward) # 先归一化 在FFN 再ResidualConnection
        return x
    
# 8. Encoder 构建编码器
class Encoder(nn.Module):
    
    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

# 9. DeocoderBlock 构建Decoder解码器的最小组成单元unit
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))  # need to mask   mask multi-head attention
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # x as the query , encoder_output from encoder
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# 10. Build the Decoder composed of DecoderBlock
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# 11. ProjectLayer() 模型输出的最后一层 用于将 (batch_size, seq_len, d_model) -> (batch_size,seq_len,vocab_size)
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

# 12. Build the whole transformer model
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src) # src is the src_input
        src = self.src_pos(src)
        return self.encoder(src, src_mask) # whether using the src mask
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt) #  tgt is the tgt_input that a kind of language is different from the input_src
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) # probably need to use mask
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

# factory_pattern 采用工厂模式 产生一个transformer model根据需要
def create_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)   # add the N=6 encoder_block as the encoder

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)  # add the N=6 decoder_block as the decoder
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer