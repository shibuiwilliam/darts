import os
import sys
import time
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import torch
import utils
import logging
import argparse
import math
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
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
parser.add_argument('--device_latency', action='store_true', default=False, help='get loss with device latency regarded')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

with open('./performance.json') as f:
    LUT = json.load(f)

C_dict = {
    0:16, # normal
    1:16, # normal
    2:32, # reduction
    3:32, # normal
    4:32, # normal
    5:64, # reduction
    6:64, # normal
    7:64 # normal
}
Alpha = 0.2
Beta = 0.6
def get_latency_coeff(z,a=Alpha,b=Beta):
    return (a*math.log(z))**b

_w = []
for _z in  [x for x in range(100000, 350000, 25000)]:
    _w.append(get_latency_coeff(_z,Alpha,Beta))
    
__w = np.array([[w_] for w_ in _w])
SCALER = MinMaxScaler(feature_range=(0.5,3.0))
SCALER.fit(__w)

def get_total_latency(genotype,lut=LUT,c_dict=C_dict):
    estimated_total_ave_second = 0
    if args.device_latency:
        genotypes = [
            [g[0] for g in genotype.normal_0],
            [g[0] for g in genotype.normal_1],
            [g[0] for g in genotype.reduce_2],
            [g[0] for g in genotype.normal_3],
            [g[0] for g in genotype.normal_4],
            [g[0] for g in genotype.reduce_5],
            [g[0] for g in genotype.normal_6],
            [g[0] for g in genotype.normal_7],
        ]
        for k,v in C_dict.items():
            for g in genotypes[k]:
                estimated_total_ave_second += LUT[g][str(v)]["average_second"]
        return estimated_total_ave_second
    else:
        genotype_normal = [g[0] for g in genotype.normal]
        genotype_reduce = [g[0] for g in genotype.reduce]
        for k,v in C_dict.items():
            for n,r in zip(genotype_normal, genotype_reduce):
                if k == 2 or k == 5:
                    estimated_total_ave_second += LUT[r][str(v)]["average_second"]
                else:
                    estimated_total_ave_second += LUT[n][str(v)]["average_second"]
        return estimated_total_ave_second
    
CIFAR_CLASSES = 10

estimated_total_ave_micro_second_list = []
latency_coef_list = []

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
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, d=args.device_latency)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, 
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=False, 
      num_workers=4)

  valid_queue = torch.utils.data.DataLoader(
      train_data, 
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=False, 
      num_workers=4)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)
    
    logging.info("Device latency: {0}".format(args.device_latency))
    if args.device_latency:
        logging.info("Multiply loss with device latency coefficient.")
    estimated_total_ave_second = get_total_latency(genotype=genotype)
    estimated_total_ave_micro_second = estimated_total_ave_second * 1000000
    estimated_total_ave_micro_second_list.append(estimated_total_ave_micro_second)
    latency_coef = float(SCALER.transform(np.array([[get_latency_coeff(estimated_total_ave_micro_second)]]))[0][0])
    latency_coef_list.append(latency_coef)
    
    logging.info('Estimated total average micro seconds per prediction: {0}\tcoeff: {1}'.format(estimated_total_ave_micro_second, latency_coef))
    for i,s in enumerate(estimated_total_ave_micro_second_list):
        logging.info("Record_{0}: {1}\t{2}".format(i,s,latency_coef_list[i]))

    archs = model.arch_parameters()
    for v in archs:
        logging.info(F.softmax(v, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, latency_coef)
    logging.info('TRAIN accuracy: %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, latency_coef)
    logging.info('VALID accuracy: %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, latency_coef):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    if args.device_latency:
        loss = loss * latency_coef

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('TRAIN: steps: %03d loss.avg: %e top1.avg: %f top5.avg: %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, latency_coef):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)
        if args.device_latency:
            loss = loss * latency_coef

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('VALID: steps: %03d loss.avg: %e top1.avg: %f top5.avg: %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

