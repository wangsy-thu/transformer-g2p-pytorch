# transformer [Attention is all your need]
# Encoder/Decoder/MHA/PFF/Transformer
import torch
from torch import nn
from config import HP
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))

        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(HP.grapheme_size, HP.encoder_dim)
        self.pe = PositionalEncoding(d_model=HP.encoder_dim, max_len=HP.encoder_max_input)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(HP.encoder_layer)])
        self.drop = nn.Dropout(HP.encoder_drop_prob)
        self.register_buffer('scale', torch.sqrt(torch.tensor(HP.encoder_dim).float()))

    def forward(self, inputs, inputs_mask):
        # inputs shape: [N, max_seq_len]

        token_emb = self.token_embedding(inputs)  # token(inputs) embedding [N, max_seq_len, en_dim]
        inputs = self.pe(token_emb * self.scale)  # positional encoding [N, max_seq_len, en_dim]
        inputs = self.drop(inputs)  # [N, max_seq_len, en_dim]

        # loop encoder layer: 6 layers
        for idx, layer in enumerate(self.layers):
            inputs = layer(inputs, inputs_mask)

        return inputs  # [N, max_seq_len, en_dim]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.self_att_layer_norm = nn.LayerNorm(HP.encoder_dim)
        self.pff_layer_norm = nn.LayerNorm(HP.encoder_dim)

        self.self_att = MultiHeadAttentionLayer(HP.encoder_dim, HP.nhead)
        self.pff = PointWiseFeedForwardLayer(HP.encoder_dim, HP.encoder_feed_forward_dim, HP.feed_forward_drop_prob)

        self.dropout = nn.Dropout(HP.encoder_drop_prob)

    def forward(self, inputs, inputs_mask):
        # inputs shape: [N, max_seq_len, en_dim]
        _inputs, att_res = self.self_att(inputs, inputs, inputs, inputs_mask)  # [N, max_seq_len, en_dim]
        inputs = self.self_att_layer_norm(inputs + self.dropout(_inputs))  # [N, max_seq_len, en_dim]
        _inputs = self.pff(inputs)
        inputs = self.pff_layer_norm(inputs + self.dropout(_inputs))
        return inputs  # [N, max_seq_len, en_dim]


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, nhead):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hid_dim = hid_dim  # 128
        self.nhead = nhead  # 2, 4, 8, 9(x)
        assert not self.hid_dim % self.nhead
        self.head_dim = self.hid_dim // self.nhead

        # Q K V input linear layer
        self.fc_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_v = nn.Linear(self.hid_dim, self.hid_dim)

        self.fc_o = nn.Linear(self.hid_dim, self.hid_dim)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.hid_dim).float()))

    # query: # [N, max_seq_len_q, en_dim]
    # key:   # [N, max_seq_len_k, en_dim]
    # value: # [N, max_seq_len_v, en_dim]
    def forward(self, query, key, value, inputs_mask=None):
        bn = query.size(0)
        # input linear layer
        Q = self.fc_q(query)  # [N, max_seq_len_q, en_dim]
        K = self.fc_k(key)  # [N, max_seq_len_k, en_dim]
        V = self.fc_v(value)  # [N, max_seq_len_v, en_dim]

        # split into nhead: 128 hid_dim, 4 heads, -> head_dim = 128/4=32
        # [N, nhead, max_seq_len_q, head_dim]
        Q = Q.view(bn, -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        # [N, nhead, max_seq_len_k, head_dim]
        K = K.view(bn, -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        # [N, nhead, max_seq_len_v, head_dim]
        V = V.view(bn, -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))

        # Energy calc
        # Q: [N, nhead, max_seq_len_q, head_dim]
        # K^T: [N, nhead, head_dim, max_seq_len_k]
        # energy: [N, nhead, max_seq_len_q, max_seq_len_k]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if inputs_mask is not None:
            """
            [[True, True, 7, 12, 2],
             [1, 3, 5, 2, False]
             [1, 3, 2, False, False]
             ....
            ]
            """
            energy = energy.masked_fill(inputs_mask == 0, -1.e10)
        # attention:[N, nhead, max_seq_len_q, max_seq_len_k]
        attention = F.softmax(energy, dim=-1)

        # attention:[N, nhead, max_seq_len_q, max_seq_len_k]
        # V:        [N, nhead, max_seq_len_v, head_dim]
        # max_seq_len_v = max_seq_len_k
        # out:      [N, nhead, max_seq_len_q, head_dim]
        out = torch.matmul(attention, V)
        # out.permute((0, 2, 1, 3)): [N, max_seq_len_q, nhead, head_dim]
        out = out.permute((0, 2, 1, 3)).contiguous()  # memory layout
        out = out.view((bn, -1, self.hid_dim))  # [N, max_seq_len_q, hid_dim]
        out = self.fc_o(out)
        return out, attention  # [N, max_seq_len_q, hid_dim] 'jack' -> "JH AE1 K"


class PointWiseFeedForwardLayer(nn.Module):
    def __init__(self, hid_dim, pff_dim, pff_drop_out):
        super(PointWiseFeedForwardLayer, self).__init__()
        self.hid_dim = hid_dim
        self.pff_dim = pff_dim
        self.pff_drop_out = pff_drop_out

        self.fc1 = nn.Linear(self.hid_dim, self.pff_dim)
        self.fc2 = nn.Linear(self.pff_dim, self.hid_dim)

        self.dropout = nn.Dropout(self.pff_drop_out)

    def forward(self, inputs):  # inputs: [N, max_seq_len, hid_dim]
        inputs = self.dropout(F.relu(self.fc1(inputs)))  # [N, max_seq_len, pff_dim]
        out = self.fc2(inputs)  # [N, max_seq_len, hid_dim]
        return out  # [N, max_seq_len, hid_dim]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(HP.phoneme_size, HP.decoder_dim)
        self.pe = PositionalEncoding(d_model=HP.decoder_dim, max_len=HP.MAX_DECODE_STEP)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(HP.decoder_layer)])
        self.fc_out = nn.Linear(HP.decoder_dim, HP.phoneme_size)  # in_feature: 128, phoneme_size
        self.drop = nn.Dropout(HP.decoder_drop_prob)
        self.register_buffer('scale', torch.sqrt(torch.tensor(HP.decoder_dim).float()))

    def forward(self, trg, enc_src, trg_mask, src_mask):
        token_emb = self.token_embedding(trg)
        pos_emb = self.pe(token_emb * self.scale)
        trg = self.drop(pos_emb)  # [N, max_seq_len, de_dim]

        for idx, layer in enumerate(self.layers):
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        out = self.fc_out(trg)  # [N, seq_len, phoneme_size]
        return out, attention


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mask_self_att = MultiHeadAttentionLayer(HP.decoder_dim, HP.nhead)
        self.mask_self_norm = nn.LayerNorm(HP.decoder_dim)

        self.mha = MultiHeadAttentionLayer(HP.decoder_dim, HP.nhead)
        self.mha_norm = nn.LayerNorm(HP.decoder_dim)

        self.pff = PointWiseFeedForwardLayer(HP.decoder_dim, HP.decoder_feed_forward_dim, HP.feed_forward_drop_prob)
        self.pff_norm = nn.LayerNorm(HP.decoder_dim)

        self.dropout = nn.Dropout(HP.decoder_drop_prob)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.mask_self_att(trg, trg, trg, trg_mask)  # self attention 不仅要mask padding, 还要保证解码t时刻，只能看到t和t之前的时间步
        trg = self.mask_self_norm(trg + self.dropout(_trg))

        _trg, attention = self.mha(trg, enc_src, enc_src, src_mask)
        trg = self.mha_norm(trg + self.dropout(_trg))

        _trg = self.pff(trg)
        trg = self.pff_norm(trg + self.dropout(_trg))

        return trg, attention  # attention: plot attention map


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def create_src_mask(src):
        """
            [[1, 23, 7, 12, 2],
             [1, 3, 5, 2, 0]
             [1, 3, 2, 0, 0]
             ....
            ]
        """
        mask = (src != HP.ENCODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device)
        return mask

    @staticmethod
    def create_trg_mask(trg):  # trg shape: [N, max_seq_len]
        trg_len = trg.size(1)
        pad_mask = (trg != HP.DECODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device)
        sub_mask = torch.tril(torch.ones(size=(trg_len, trg_len), dtype=torch.uint8)).bool().to(HP.device)
        trg_mask = pad_mask & sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.create_src_mask(src)
        # print("src mask", torch.tensor(src_mask, dtype=torch.uint8))
        trg_mask = self.create_trg_mask(trg)
        # print("trg mask", torch.tensor(trg_mask, dtype=torch.uint8))

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention  # output [N, seq_len, phoneme_size]

    def infer(self, x):  # word: jack [[1, 25, 6, 2]] -> "JH AE1 K"
        batch_size = x.size(0)
        src_mask = self.create_src_mask(x)  # create src(word) mask
        enc_src = self.encoder(x, src_mask)  # encoder output

        # init trg shape: [1, 1]
        trg = torch.zeros(size=(batch_size, 1)).fill_(HP.DECODER_SOS_IDX).long().to(HP.device)
        decoder_step = 0
        while True:
            if decoder_step == HP.MAX_DECODE_STEP:
                print("Warning: Reached Max Decoder Step")
                break
            trg_mask = self.create_trg_mask(trg)  # create target mask
            output, attention = self.decoder(trg, enc_src, trg_mask,
                                             src_mask)  # output shape: [1, seq_len, phoneme_size]
            pred_token = output.argmax(-1)[:, -1]  # shape : [1,]
            trg = torch.cat((trg, pred_token.unsqueeze(0)), dim=-1)  # trg [1, 2]
            if pred_token.item() == HP.DECODER_EOS_IDX:
                print('Decode Done!')
                break
            decoder_step += 1
        return trg[:, 1:], attention


# class Transformer(nn.Module):
#     def __init__(self,):
#         super(Transformer, self).__init__()
#
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#
#     @staticmethod
#     def create_src_mask(src):
#         # src: [N, seq_len]
#         # mask: [N, 1, 1, src_len]
#         # energy: [N, nhead, max_seq_len_q, max_seq_len_k]
#         mask = (src != HP.ENCODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device)
#         return mask
#
#     @staticmethod
#     def create_trg_mask(trg):
#         trg_len = trg.size(1)
#         pad_mask = (trg != HP.DECODER_PAD_IDX).unsqueeze(1).unsqueeze(2).to(HP.device)
#         sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8)).bool().to(HP.device)
#         # trg_mask: [bn, 1, trg_len, trg_len]
#         trg_mask = pad_mask & sub_mask
#         return trg_mask
#
#     def forward(self, src, trg):
#         # src_mask: [bn, 1, 1, src_len]
#         src_mask = self.create_src_mask(src)
#         # trg_mask: [bn, 1, trg_len, trg_len]
#         trg_mask = self.create_trg_mask(trg)
#         # # enc_src = [batch size, src len, hid dim]
#         enc_src = self.encoder(src, src_mask)
#         #
#         # # output: [bn, trg_len, phoneme_size]
#         # # attention: [bn, 1nheads, trg_src, src_len]
#         output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
#         return output, attention
#
#     def infer(self, x):
#         batch_size = x.size(0)
#         src_mask = self.create_src_mask(x)
#         enc_src = self.encoder(x, src_mask)
#
#         trg = torch.zeros((batch_size, 1)).fill_(self.hp.DECODER_SOS_IDX).long().to(self.hp.device)
#         decoder_step = 0
#         decode_done_dict = dict(enumerate([0]*batch_size))
#         while True:
#             if 0 not in list(decode_done_dict.values()):  # all decode to <eos>
#                 break
#             if decoder_step == self.hp.decoder_max_output:
#                 print("Warming: Reached max decoder steps")
#                 break
#             trg_mask = self.create_trg_mask(trg)
#             output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
#
#             # [bn, 1]
#             pred_token = output.argmax(-1)[:, -1].long()  # only get the final one
#             check_list = pred_token.data.cpu().numpy().tolist()
#             for idx, v in enumerate(check_list):
#                 if v == self.hp.DECODER_EOS_IDX and (not decode_done_dict[idx]):
#                     decode_done_dict[idx] = decoder_step
#             pred_token = pred_token.unsqueeze(1)
#             trg = torch.cat((trg, pred_token), dim=-1)
#             decoder_step += 1
#
#         att_sum = torch.sum(attention, dim=1).squeeze()
#         return trg[:, 1:], decode_done_dict, att_sum # torch.argmax(att_sum, dim=0).cpu().tolist()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataset_g2p import G2PDataset, collate_fn

    net = Transformer()
    test_set = G2PDataset('./data/data_test.json')
    test_loader = DataLoader(test_set, batch_size=3, collate_fn=collate_fn)
    for batch in test_loader:
        words_idx, words_len, phoneme_seqs_idx, phoneme_len = batch
        print('words_idx', words_idx.size())
        print('phoneme_seqs_idx', phoneme_seqs_idx.size())
        pout, att = net(words_idx, phoneme_seqs_idx)
        break
