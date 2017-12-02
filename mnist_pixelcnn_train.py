import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.mnist_dataset import MnistDataset
from data.data_affine import RandomAffineTransform
from models.mnist_pixelcnn import *
from models.components.shared import *
from train_utils import *

def quantisize(images, levels):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')

class Evaluator:
    def __init__(self, args, model, ds=None, batch_size=100):
        self.model = model
        self.args = args
        if ds is None:
            ds = MnistDataset(transform=[], train=False)
        self.ds = ds
        self.batch_size = batch_size
        self.loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
            shuffle=False, num_workers=args.workers, drop_last=False)

    def evaluate(self, show_prog=False):
        nll_total = 0
        Ntotal = 0
        if show_prog:
            print("Evaluating...")
            prog = tqdm.tqdm(total=len(self.ds))
        for i, (labels, x, _) in enumerate(self.loader, 0):
            N, labels, x, x_quant = preprocess(self.args, self.model, labels, x)
            dist = self.model(x, labels=labels)
            nll = -dist.log_prob(x_quant)
            nll_total = nll_total + nll.data
            Ntotal = Ntotal + N
            if show_prog:
                prog.update(N)
        if show_prog:
            prog.close()
        nll = nll_total / Ntotal
        return nll

def preprocess(args, model, labels, x):
    x_quant = model.quant(x, args.levels)
    x = model.dequant(x_quant, args.levels)
    #x_quant = torch.from_numpy(quantisize(x.numpy(),
    #    args.levels)).type(torch.LongTensor)
    #x = x_quant.float() / (args.levels - 1)
    if args.gpu:
        x = x.cuda()
        x_quant = x_quant.cuda()
        labels = labels.cuda()
    x = Variable(x, requires_grad=False)
    x_quant = Variable(x_quant, requires_grad=False)
    N = x.size(0)
    labels = Variable(labels, requires_grad=False)
    return N, labels, x, x_quant

if __name__ == "__main__":
    parser = ArgParser(description='Train pixelcnn on MNIST')
    parser.add_argument('--conditional', action='store_true',
            help='conditioned on digit labels')
    parser.add_argument('--levels', type=int, default=8,
            help='levels for quantisization')
    parser.add_argument('--layers', type=int, default=5,
            help='layers')
    parser.add_argument('--hidden-dims', type=int, default=32,
            help='hidden dimensions')
    args = parser.parse_args()

    model = MNIST_PixelCNNNew(levels=args.levels,
            layers=args.layers,
            conditional=args.conditional,
            hidden_dims=args.hidden_dims)
    if args.gpu:
        model = model.cuda()

    ds = MnistDataset(transform=[])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=args.gpu,
            drop_last=False)

    evaluator = Evaluator(args, model)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    trainer = Trainer(args, model, optimizer)

    for epoch in range(trainer.start_epoch, args.epochs):
        trainer.update_lr(epoch)
        print_now('epoch=%d --------------------------------' % epoch)
        evaluation_nll = evaluator.evaluate(show_prog=True)
        print("Epoch %d evaluation NLL: %.f" % (epoch, evaluation_nll))
        if trainer.writer is not None:
            trainer.writer.add_scalar('loss/eval-nll', evaluation_nll, trainer.step)
        for i, (labels, x, _) in enumerate(loader, 0):
            N, labels, x, x_quant = preprocess(args, model, labels, x)
            dist = model(x, labels=labels)
            nll = -dist.log_prob(x_quant) / N
            loss = nll
            trainer.loss_update(loss.data.cpu().numpy())
            if i % args.print_every == 0:
                print("  epoch %d, step %d. Loss ema: %.3f" % (epoch, i, trainer.loss)) 
                if trainer.writer is not None:
                    #trainer.writer.add_scalar('loss/log_prob_x', log_prob_x, trainer.step)
                    trainer.writer.add_scalar('loss/loss', loss, trainer.step)
                    trainer.writer.add_scalar('loss/loss-ema', trainer.loss, trainer.step)
            trainer.minimize(loss)
        if trainer.writer is not None:
            new_labels = torch.LongTensor([0, 1, 2])
            if labels.data.is_cuda:
                new_labels = new_labels.cuda()
            new_labels = Variable(new_labels, requires_grad=False)
            x_samp = model.generate_samples(28, 28, 1, 3, labels=new_labels)
            print("samp stats: ", x_samp.max(), x_samp.min(), x_samp.mean())
            if x_samp is not None:
                for ex in range(3):
                    trainer.writer.add_image('x-samples/samp%d' % ex, torch_bw_img_to_np_img(x_samp[ex]), trainer.step)

            mlp = model.dequant(dist.MAP, args.levels)
            for ex in range(3):
                trainer.writer.add_image('x-train/mlp-%d' % ex, torch_bw_img_to_np_img(mlp[ex]), trainer.step)
                trainer.writer.add_image('x-train/input-%d' % ex, torch_bw_img_to_np_img(x[ex]), trainer.step)
        trainer.save_checkpoint(epoch + 1)
