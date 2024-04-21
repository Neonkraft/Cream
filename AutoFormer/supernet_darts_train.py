import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import wandb

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from model.supernet_transformer import Vision_TransformerSuper, wrap_entangled_modules

from model import utils
from architect import Architect

DEBUG_MODE = False

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--lora_rank', type=int, default=4, help='rank of lora layers')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=5e-4, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=10, help='num of warmup training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--lr-power', type=float, default=1.0,
                    help='power of the polynomial lr scheduler')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


args = parser.parse_args()

# stall for a random amount of time to avoid multiple processes logging at the same time
time.sleep(np.random.uniform(0, 5))

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def get_config_for_space(space):
    if space == 'tiny':
        max_embed_dim = 240
        max_mlp_ratio = 4
        max_num_heads = 4
        max_depth = 14

        # choices
        embed_choices = [24, 192, 240]
        mlp_ratio_choices = [0.5, 3.5, 4]
        n_heads_choices = [1, 3, 4]
        n_layer_choices = [1, 12, 14]
    elif space == 'small':
        max_embed_dim = 448
        max_mlp_ratio = 4
        max_num_heads = 7
        max_depth = 14

        # choices
        embed_choices = [64, 320, 448]
        mlp_ratio_choices = [0.5, 3, 4]
        n_heads_choices = [1, 5, 7]
        n_layer_choices = [1, 12, 14]
    else:
        raise ValueError('Invalid version {version}')

    biggest_config = {
        "embed_dim": [max_embed_dim] * max_depth,
        "mlp_ratio": [max_mlp_ratio] * max_depth,
        "num_heads": [max_num_heads] * max_depth,
        "layer_num": max_depth,
    }

    config_options = {
        "embed_dim": embed_choices,
        "mlp_ratio": mlp_ratio_choices,
        "num_heads": n_heads_choices,
        "layer_num": n_layer_choices,
    }

    return biggest_config, config_options

def init_supernet(img_size, num_classes, space='tiny'):
    biggest_config, config_options = get_config_for_space(space)
    model = Vision_TransformerSuper(
       img_size=img_size,
       num_classes=num_classes,
       change_qkv=True,
       embed_dim=max(config_options["embed_dim"]),
       num_heads=max(config_options["num_heads"]),
       depth=max(config_options["layer_num"]),
    )
    model.set_sample_config(biggest_config)

    embed_dim_weights = nn.Parameter(torch.zeros(len(config_options["embed_dim"])))
    mlp_ratio_weights = nn.Parameter(torch.zeros(biggest_config["layer_num"], len(config_options["mlp_ratio"])))
    n_heads_weights = nn.Parameter(torch.zeros(biggest_config["layer_num"], len(config_options["num_heads"])))
    n_layers_weights = nn.Parameter(torch.zeros(len(config_options["layer_num"])))

    model.set_arch_choices(config_options['embed_dim'], config_options['mlp_ratio'], config_options['num_heads'], config_options['layer_num'])
    model.set_arch_weights(embed_dim_weights, mlp_ratio_weights, n_heads_weights, n_layers_weights)

    model = wrap_entangled_modules(model, config_options)
    return model


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = init_supernet(img_size=32, num_classes=10, space="tiny").cuda()
  model._criterion = criterion

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # optimizer = torch.optim.SGD(
  #     model.parameters(),
  #     args.learning_rate,
  #     momentum=args.momentum,
  #     weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)


  optimizer = create_optimizer(args, model)
  scheduler, _ = create_scheduler(args, optimizer)
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
  #       optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):

    if epoch == args.warmup_epochs:
      logging.info('warmup done')
      model.activate_lora(args.lora_rank)
      lora_params = [p for name, p in model.named_parameters() if "lora_" in name]
      optimizer.add_param_group({'params': lora_params, 'lr': args.lr, 'weight_decay': args.weight_decay})
      logging.info(f"lora layers activated with rank {args.lora_rank}")
      logging.info(f"Number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    scheduler.step(epoch)
    lr = scheduler._get_lr(epoch)[0]
    logging.info('epoch %d lr %e', epoch, lr)

    arch_weights = {k: F.softmax(v) for k, v in model.arch_weights.items()}
    logging.info(f"current arch parameters (post softmax): {arch_weights}")

    # training
    start_time = time.time()
    train_acc, train_acc_top5, train_obj = train(epoch, args.warmup_epochs, train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    epoch_time = time.time() - start_time
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    wandb.log({
        "train/acc1": train_acc,
        "train/acc5": train_acc_top5,
        "train/loss": train_obj,
        "valid/acc1": valid_acc,
        "valid/acc5": valid_acc_top5,
        "valid/loss": valid_obj,
        "lr": lr,
        "epoch": epoch,
        "epoch_time": epoch_time,
    })

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(epoch, warmup_epochs, train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    if epoch > warmup_epochs:
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda()
      target_search = target_search.cuda()

      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    if DEBUG_MODE:
      break

  return top1.avg, top5.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    if DEBUG_MODE:
      break

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  wandb.init(project='LoRA', config=args, name='AutoFormer-CF10')

  main()

  wandb.finish()