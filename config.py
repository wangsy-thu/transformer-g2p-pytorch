from symbols import graphemes_char2id, phonemes_char2id


class Hyperparameter:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cpu'  # cuda training, inference cpu
    data_dir = './data/'
    train_dataset_path = './data/data_train.json'
    val_dataset_path = './data/data_val.json'
    test_dataset_path = './data/data_test.json'
    seed = 123
    # ################################################################
    #                             Model Structure
    # ################################################################
    # encoder
    encoder_layer = 6  # encoder layer number, default 6 layers
    encoder_dim = 128  # token embedding and transformer hidden dimension, decoder dim, positional encode
    encoder_drop_prob = 0.1
    grapheme_size = len(graphemes_char2id)
    encoder_max_input = 30

    # multi head hp
    nhead = 4  # self attention head number

    # feed-forward layer: linear layer -> conv1d -> conv1d + DS
    encoder_feed_forward_dim = 1024
    decoder_feed_forward_dim = 1024
    feed_forward_drop_prob = 0.3

    # decoder
    decoder_layer = 6
    decoder_dim = encoder_dim
    decoder_drop_prob = 0.1
    phoneme_size = len(phonemes_char2id)
    MAX_DECODE_STEP = 50

    ENCODER_SOS_IDX = graphemes_char2id['<s>']
    ENCODER_EOS_IDX = graphemes_char2id['</s>']
    ENCODER_PAD_IDX = graphemes_char2id['<pad>']
    DECODER_SOS_IDX = phonemes_char2id['<s>']
    DECODER_EOS_IDX = phonemes_char2id['</s>']
    DECODER_PAD_IDX = phonemes_char2id['<pad>']

    # ################################################################
    #                             Experiment
    # ################################################################
    batch_size = 128
    init_lr = 1e-4
    epochs = 100
    verbose_step = 100
    save_step = 500
    grad_clip_thresh = 1.


HP = Hyperparameter()
