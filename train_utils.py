import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from models.components.shared import *
import os
import json
import shutil
import sys
from tensorboardX import SummaryWriter
import numpy as np

class ArgParser(argparse.ArgumentParser):
    def __init__(self, description='Train'):
        super(ArgParser, self).__init__(description=description)
        parser = self
        parser.add_argument('-b', '--batch-size', metavar='BATCH', type=int, default=128,
                help='batch size for training')
        parser.add_argument('-j', '--workers', metavar='NWORKERS', type=int, default=1,
                help='number of dataloader workers (default 1)')
        parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=300,
                help='number of epochs to train (default 300)')
        parser.add_argument('-o', '--expdir', metavar='EXPDIR', type=str,
                default=os.environ.get("OUTPUT_DIR", None),
                help='output directory for model checkpoints and tensorboard summaries')
        parser.add_argument('-p', '--print-every', metavar='PRINT_FREQ', type=int,
                default=10, help='print frequency (steps)')
        parser.add_argument('-s', '--sample-every', metavar='SAMPLE_FREQ', type=int,
                default=50, help='sample frequency (steps)')
        parser.add_argument('--lr', metavar='LEARNING_RATE', type=float,
                default=1e-4, help='learning rate')
        parser.add_argument('--lr-decay', metavar='LEARNING_RATE_DECAY', type=float,
                default=0.9, help='learning rate decay')
        parser.add_argument('--decay-every', metavar='DECAY_EVERY', type=int,
                default=1, help='decay every')
        parser.add_argument('--no-gpu', dest='disable_gpu', action='store_true', 
                help='disable use of GPU')
        parser.add_argument('--resume', type=str, default=None,
                help='resume training from given checkpoint')
        parser.add_argument('--clip-grad', type=float, default=1.0, help='clip gradient to norm (default: 1.0)')

    def parse_args(self):
        args = super(ArgParser, self).parse_args()
        args.gpu = gpu and not args.disable_gpu
        if args.expdir is not None:
            os.makedirs(args.expdir, exist_ok=True)
            with open(os.path.join(args.expdir, "args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
        args.anneal = lambda epoch: args.anneal_full * (epoch / float(args.anneal_by_epoch) if epoch < args.anneal_by_epoch else 1)
        return args

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    if args.expdir is None:
        return
    filepath = os.path.join(args.expdir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(args.expdir, 'model_best.pth.tar'))

class Trainer:
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.start_epoch = 0
        self.step = 0
        self.loss_ema = EMA(decay=0.95)
        self.checkpoint = None
        self.writer = None
        if self.args.expdir is not None:
            self.writer = SummaryWriter(log_dir=self.args.expdir)
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint %s" % self.args.resume)
                self.checkpoint = torch.load(self.args.resume)
                self.start_epoch = self.checkpoint['epoch']
                self.step = self.checkpoint['step']
                self.model.load_state_dict(self.checkpoint['state_dict'])
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                print("=> loaded checkpoint %s (epoch %d)" % (self.args.resume, self.start_epoch))

    def update_lr(self, epoch):
        torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                self.args.lr_decay,
                last_epoch=(epoch // self.args.decay_every) - 1)

    def save_checkpoint(self, epochs, **kwargs):
        state = dict(
            epoch=epochs,
            step=self.step,
            model_desc=str(self.model),
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        state.update(kwargs)
        save_checkpoint(self.args, state, False)

    def loss_update(self, loss):
        self.loss_ema.update(loss)

    def minimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.clip_grad is not None and self.args.clip_grad > 1e-12:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_grad)
        self.optimizer.step()
        self.step += 1

    @property
    def loss(self):
        return self.loss_ema()

def torch_bw_img_to_np_img(img):
    img = np.tile(img.data.cpu().float().numpy(), [3, 1, 1])
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, [1, 2, 0])
    return img

def print_now(string):
    sys.stdout.write("%s\n" % string)
    sys.stdout.flush()
