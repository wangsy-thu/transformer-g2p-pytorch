from torch.utils.data import Dataset
import json
from symbols import word2id, phoneme2id, graphemes_char2id, phonemes_char2id
import torch


class G2PDataset(Dataset):
    def __init__(self, dataset_path):  # json path
        data_dict = json.load(open(dataset_path, 'r'))
        self.data_pairs = list(data_dict.items())  # [('jack', "JH AE1 K"), (...)]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        word, phone_seq = self.data_pairs[idx][0], self.data_pairs[idx][1]
        return word2id(word), phoneme2id(phone_seq)  # to idx ([23,5, 21], [30,54,12])


def collate_fn(iter_batch):  # pad & descending sort & to tensor
    N = len(iter_batch)  # batch size
    word_indexes, phoneme_indexes = [list(it) for it in zip(*iter_batch)]

    # add start & end token for both word and phoneme
    # input start token
    [it.insert(0, graphemes_char2id['<s>']) for it in word_indexes]  # [12, 26, 29]
    [it.append(graphemes_char2id['</s>']) for it in word_indexes]   # [1, 12, 26, 29, 2]

    # MUST! output start and end token
    [it.insert(0, phonemes_char2id['<s>']) for it in phoneme_indexes]  # [12, 26, 29]
    [it.append(phonemes_char2id['</s>']) for it in phoneme_indexes]  # [1, 12, 26, 29, 2]

    # descending sort input sequence: [19,18,5...], [1, 0, 2,...]
    word_lengths, sort_idx = torch.sort(torch.tensor([len(it) for it in word_indexes]).long(), descending=True)
    max_word_len = word_lengths[0]
    word_padded = torch.zeros((N, max_word_len)).long()  # shape: [N, max_seq_len]

    max_phoneme_len = max([len(it) for it in phoneme_indexes])
    phoneme_padded = torch.zeros((N, max_phoneme_len)).long()
    phoneme_lengths = torch.zeros((N,)).long()

    for idx, idx_s in enumerate(sort_idx.tolist()):
        word_padded[idx][:word_lengths[idx]] = torch.tensor(word_indexes[idx_s]).long()
        phoneme_padded[idx][:len(phoneme_indexes[idx_s])] = torch.tensor(phoneme_indexes[idx_s]).long()
        phoneme_lengths[idx] = len(phoneme_indexes[idx_s])

    return word_padded, word_lengths, phoneme_padded, phoneme_lengths


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    val_set = G2PDataset('./data/data_val.json')
    val_loader = DataLoader(val_set, batch_size=7, collate_fn=collate_fn)
    for batch in val_loader:
        words_idx, words_len, phoneme_seq_idx, phoneme_len = batch
        print('grapheme batch tensor size', words_idx.size())
        print(words_idx)
        print('grapheme lengths: ', words_len)
        print('*'*88)
        print('phoneme batch tensor size', phoneme_seq_idx.size())
        print(phoneme_seq_idx)
        print('phoneme lengths: ', phoneme_len)
        break
