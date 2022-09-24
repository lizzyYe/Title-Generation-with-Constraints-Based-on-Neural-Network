import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            # >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_head, num_encoder, num_decoder, dim_feed,dropout=0.1):#embedding_dim, hidden_dim, num_layers, p_dropout,num_class,bidic
        super(TransformerModel, self).__init__()
        self.d_model = embedding_dim

        # self.bn=nn.BatchNorm1d(embedding_dim)
        # self.bn_2=nn.BatchNorm1d(vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_tgt = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = PositionalEncoding(embedding_dim)
        self.pos_emb_tgt = PositionalEncoding(embedding_dim)
        self.trans = nn.Transformer(d_model=embedding_dim, nhead=n_head, num_encoder_layers=num_encoder,
                                    num_decoder_layers=num_decoder,dim_feedforward= dim_feed,dropout=dropout)
        self.Linear = nn.Linear(embedding_dim,vocab_size)


    def _generate_square_subsequent_mask(self, sz):
        '''
        sequence_mask for decoder
        :param sz:
        :return:
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # def padding_mask(self,seq_k, seq_q):
    #     '''
    #     :param seq_k: batch_size*key_seq_len
    #     :param seq_q: batch_size*query_seq_len
    #     :return:
    #     '''
    #     # seq_k和seq_q的形状都是[B,L]
    #     len_q = seq_q.size(1)
    #     # `PAD` is 0
    #     pad_mask = seq_k.eq(0)
    #     pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    #     return pad_mask



    def forward(self, src, tgt,device='cuda'):
        '''

        :param src: batch_size*src_seq_len
        :param tgt: batch_size*tgt_seq_len
        :return:
        '''

        src_embbed = self.embedding(src.transpose(0,1))*math.sqrt(self.d_model)
        src_embbed = self.pos_emb(src_embbed).to(device)

        tgt_embbed = self.embedding_tgt(tgt.transpose(0, 1))*math.sqrt(self.d_model)
        tgt_embbed = self.pos_emb_tgt(tgt_embbed).to(device)

        # src_embbed =self.bn(src_embbed.transpose(1,2)).transpose(1,2)
        # tgt_embbed =self.bn(tgt_embbed.transpose(1,2)).transpose(1,2)

        # src_embbed: src_seq_len*batch size*embed dim
        # tgt_embbed: tgt_seq_len*batch size*embed dim

        # mask
        # src_key_padding_mask = src.eq(0).to(device)
        # tgt_key_padding_mask = tgt.eq(0).to(device)
        # src_key_padding_mask: batch size*src_seq_len
        # tgt_key_padding_mask: batch size*tgt_seq_len

        src_mask = self._generate_square_subsequent_mask(src_embbed.size(0)).to(device)
        tgt_mask = self._generate_square_subsequent_mask(tgt_embbed.size(0)).to(device)

        output = self.trans(src_embbed,tgt_embbed,src_mask,tgt_mask=tgt_mask)
        # output = self.trans(src_embbed,tgt_embbed,src_mask,tgt_mask,memory_mask=None, src_key_padding_mask=src_key_padding_mask,
        #                     tgt_key_padding_mask=tgt_key_padding_mask)

        output = self.Linear(output)
        # output = self.dropout(output)
        # output=self.bn_2(output.transpose(1,2)).transpose(1,2)

        if self.training:
            output = F.log_softmax(output, dim=-1).transpose(0, 1)
        else:
            output = F.softmax(output, dim=-1).transpose(0, 1)



        return output

