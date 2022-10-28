# 1. g2p
# 2. plot attention map
from model import Transformer
import torch
from config import HP
from symbols import word2id, id2phoneme
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # openKMP

HP.device = 'cpu'
# new and load model
model = Transformer()
checkpoint = torch.load('./model_save/model_65_45000.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

while 1:
    word = input("Input: ").strip()  # digital
    wordids = word2id(word.lower())  # jack -> [3, 7, 13, 8]
    wordids = [HP.ENCODER_SOS_IDX] + wordids + [HP.ENCODER_EOS_IDX]  # [1, 3, 7, 13, 8, 2]
    wordids_t = torch.tensor(wordids).unsqueeze(0)  # src input: [N, max_seq_len]=[1, 6]
    phonemes, attention_weight = model.infer(wordids_t)
    phoneme_list = phonemes.squeeze().cpu().numpy().tolist()
    phoneme_seq = id2phoneme(phoneme_list)
    print(phoneme_seq)
    print(attention_weight.size())
    word_tokens = ['<s>'] + list(word.lower()) + ['</s>']
    phoneme_tokens = phoneme_seq.split(' ')
    attention_map_weight = torch.sum(attention_weight.squeeze(), dim=0)  # attention_map_weight : [seq_phone_len,
    # seq_word_len]
    att_matrix = attention_map_weight.transpose(0, 1).detach().cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(att_matrix)

    ax.set_xticks(np.arange(len(phoneme_tokens)))
    ax.set_yticks(np.arange(len(word_tokens)))

    ax.set_xticklabels(phoneme_tokens)
    ax.set_yticklabels(word_tokens)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title("word-phoneme Attention Map")
    fig.tight_layout()
    plt.show()
