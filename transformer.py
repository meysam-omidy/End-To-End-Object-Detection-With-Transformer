import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, main_dim, num_heads, k_dim=None, v_dim=None):
        super().__init__()
        self.main_dim = main_dim
        self.num_heads = num_heads
        self.k_dim = k_dim if k_dim != None else main_dim
        self.v_dim = v_dim if v_dim != None else main_dim
        self.query = nn.Linear(self.main_dim, self.k_dim)
        self.key = nn.Linear(self.main_dim, self.k_dim)
        self.value = nn.Linear(self.main_dim, self.v_dim)
        self.fc = nn.Linear(self.v_dim, self.main_dim)

    def forward(self, input1, input2, mask=None):
        batch_size1, num_tokens1, dim1 = input1.size()
        batch_size2, num_tokens2, dim2 = input2.size()
        query = self.query(input2)
        key = self.key(input1)
        value = self.value(input1)
        query = query.view(batch_size2, num_tokens2, self.num_heads, self.k_dim//self.num_heads).permute(0,2,1,3)
        key = key.view(batch_size1, num_tokens1, self.num_heads, self.k_dim//self.num_heads).permute(0,2,1,3)
        value = value.view(batch_size1, num_tokens1, self.num_heads, self.v_dim//self.num_heads).permute(0,2,1,3)
        x = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / math.sqrt(self.k_dim//self.num_heads)
        if mask != None:
            x = x + mask[:num_tokens2]
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, value)
        x = x.permute(0,2,1,3).reshape(batch_size2, num_tokens2, -1)
        return self.fc(x)


class EncoderLayer(nn.Module):
    def __init__(self, main_dim, ff_dim, num_heads, dropout_p):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(main_dim, num_heads)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(main_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(ff_dim, main_dim)
        )
        self.norm = nn.LayerNorm(main_dim)

    def forward(self, input, mask):
        x = self.norm(input + self.multi_head_attention(input, input, mask))
        x = self.norm(x + self.positionwise_feedforward(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, main_dim, ff_dim, num_heads, dropout_p):
        super().__init__()
        self.maksked_multi_head_attention = MultiHeadAttention(main_dim, num_heads)
        self.multi_head_attention = MultiHeadAttention(main_dim, num_heads)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(main_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(ff_dim, main_dim)
        )
        self.norm = nn.LayerNorm(main_dim)

    def forward(self, input, encoder_input, mask):
        x = self.norm(input + self.maksked_multi_head_attention(input, input, mask))
        x = self.norm(x + self.multi_head_attention(encoder_input, x))
        x = self.norm(x + self.positionwise_feedforward(x))
        return x


class Transformer(nn.Module):
    def __init__(self, num_input_tokens, main_dim, ff_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout_p=0.2, max_tokens=100, 
                 num_output_tokens=None, trainable_positional_encodings=False) -> None:
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_input_tokens if num_output_tokens == None else num_output_tokens
        self.main_dim = main_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.max_tokens = max_tokens
        if trainable_positional_encodings:
            self.positional_encoding = nn.Parameter(torch.randn(max_tokens, main_dim), requires_grad=True)
        else:
            positional_encoding = torch.zeros(size=(self.max_tokens, self.main_dim)).type(torch.float32)
            for i in range(self.max_tokens):
                for j in range(self.main_dim):
                    if j%2 == 0:
                        positional_encoding[i,j] = math.sin(i/100**(2*(j//2)/self.main_dim))
                    else:
                        positional_encoding[i,j] = math.cos(i/100**(2*(j//2)/self.main_dim))
            self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=False)
        self.embedding = nn.Embedding(self.num_input_tokens, main_dim)
        self.encoder = nn.Sequential(
            *[EncoderLayer(main_dim, ff_dim, num_heads, dropout_p) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.Sequential(
            *[DecoderLayer(main_dim, ff_dim, num_heads, dropout_p) for _ in range(num_decoder_layers)]
        )
        self.fc = nn.Linear(main_dim, self.num_output_tokens)

    def forward(self, source, target, encoder_mask=None, decoder_mask=None):
        if len(source.shape) == 2:
            x = (self.embedding(source) * math.sqrt(self.main_dim)).type(torch.float32)
        else:
            x = source
        if len(target.shape) == 2:
            y = (self.embedding(target) * math.sqrt(self.main_dim)).type(torch.float32)
        else:
            y = target
        x = x + self.positional_encoding[:x.size(1)]
        y = y + self.positional_encoding[:y.size(1)]
        for module in self.encoder._modules.values():
            x = module(x, encoder_mask)
        for module in self.decoder._modules.values():
            y = module(y, x, decoder_mask)
        return self.fc(y)
    
    def sequential_forward(self, source, target, encoder_mask=None, decoder_mask=None, ratio=0.5):
        if len(source.shape) == 2:
            x = (self.embedding(source) * math.sqrt(self.main_dim)).type(torch.float32)
        else:
            x = source
        x = x + self.positional_encoding[:x.size(1)]
        for module in self.encoder._modules.values():
            x = module(x, encoder_mask)
        if len(target.shape) == 2:
            embeddings = (self.embedding(target[:, 0].unsqueeze(1)) * math.sqrt(self.main_dim)).type(torch.float32)
        else:
            embeddings = target[:, 0].unsqueeze(1)
        outputs = torch.zeros(size=(target.size(0), target.size(1), self.num_output_tokens)).to(source.device)
        for t in range(target.shape[1]):
            y = embeddings + self.positional_encoding[:embeddings.size(1)]
            for module in self.decoder._modules.values():
                if decoder_mask != None:
                    y = module(y, x, decoder_mask[:t+1, :t+1])
                else:
                    y = module(y, x, None)
            y = self.fc(y)
            outputs[:, t, :] = y[:, -1, :].squeeze(1)
            best_guess = (self.embedding(y.argmax(dim=-1)[:, -1].unsqueeze(1)) * math.sqrt(self.main_dim)).type(torch.float32)
            rands = torch.rand(size=(best_guess.size(0), 1, 1)).to(best_guess.device)
            selected = torch.where(rands > ratio, best_guess, embeddings[:, -1].unsqueeze(1))
            embeddings = torch.concatenate((embeddings, selected), dim=1)
        return outputs