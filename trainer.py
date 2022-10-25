import os
from argparse import ArgumentParser
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from model import Transformer
from config import HP
from dataset_g2p import G2PDataset, collate_fn


logger = SummaryWriter('./log')

# seed init: Ensure Reproducible Result
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)


def evaluate(model_, devloader, crit):
    model_.eval()  # set evaluation flag
    sum_loss = 0.
    with torch.no_grad():
        for batch in devloader:
            words_idxs, word_len, phoneme_seqs_idxs, phoneme_len = batch
            output_post, attention = model_(words_idxs.to(HP.device), phoneme_seqs_idxs[:, :-1].to(HP.device))
            out = output_post.view(-1, output_post.size(-1))  # [N*seq_len, phoneme_size]
            trg = phoneme_seqs_idxs[:, 1:]
            trg = trg.contiguous().view(-1)  # [N*seq_len, ]
            loss = crit(out.to(HP.device), trg.to(HP.device))
            sum_loss += loss.item()
    model_.train() # back to training mode
    return sum_loss / len(devloader)


def save_checkpoint(model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optm.state_dict()
    }
    torch.save(save_dict, checkpoint_path)


def train():
    parser = ArgumentParser(description="Model Training")
    parser.add_argument(
        '--c',
        default=None,
        type=str,
        help='train from scratch or resume training'
    )
    args = parser.parse_args()

    # new model instance
    model = Transformer()
    model = model.to(HP.device)

    # loss function (loss.py)
    criterion = nn.CrossEntropyLoss(ignore_index=HP.DECODER_PAD_IDX) # ignore PAD index

    # optimizer
    opt = optim.Adam(model.parameters(), lr=HP.init_lr)

    # train dataloader
    trainset = G2PDataset(HP.train_dataset_path)
    train_loader = DataLoader(trainset, batch_size=HP.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # dev datalader(evaluation)
    devset = G2PDataset(HP.val_dataset_path)
    dev_loader = DataLoader(devset, batch_size=HP.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

    start_epoch, step = 0, 0

    if args.c:
        checkpoint = torch.load(args.c)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Resume From %s.' % args.c)
    else:
        print('Training From scratch!')

    model.train()   # set training flag

    # main loop
    for epoch in range(start_epoch, HP.epochs):
        print('Start Epoch: %d, Steps: %d' % (epoch, len(train_loader)))
        for batch in train_loader:
            words_idxs, word_len, phoneme_seqs_idxs, phoneme_len = batch
            opt.zero_grad() # gradient clean
            # <s> JH AE1 K </s>
            # <s> JH AE1 K -> JH AE1 K </s>
            output_post, attention = model(words_idxs.to(HP.device), phoneme_seqs_idxs[:, :-1].to(HP.device))
            out = output_post.view(-1, output_post.size(-1)) # [N*seq_len, phoneme_size]
            trg = phoneme_seqs_idxs[:, 1:]
            trg = trg.contiguous().view(-1)  # [N*seq_len, ]
            loss = criterion(out.to(HP.device), trg.to(HP.device))

            loss.backward()  # backward process
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip_thresh)
            #
            opt.step()

            logger.add_scalar('Loss/Train', loss, step)

            if not step % HP.verbose_step:  # evaluate log print
                eval_loss = evaluate(model, dev_loader, criterion)
                logger.add_scalar('Loss/Dev', eval_loss, step)

            if not step % HP.save_step: # model save
                model_path = 'model_%d_%d.pth' % (epoch, step)
                save_checkpoint(model, epoch, opt, os.path.join('model_save', model_path))

            step += 1
            logger.flush()
            print('Epoch: [%d/%d], step: %d Train Loss: %.5f, Dev Loss: %.5f'
                  % (epoch, HP.epochs, step, loss.item(), eval_loss))
    logger.close()


if __name__ == '__main__':
    train()
