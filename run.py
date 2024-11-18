import argparse
import torch.nn as nn

from es import GeneticAlgorithm, load_file, load_point
from models.mobilenet import *
import conf
import os


parser = argparse.ArgumentParser()
parser.add_argument('-net', required=False, help='net type', default='mobilenet')
parser.add_argument('-epoch', help='total GA iterations', type=int, default=4000)
parser.add_argument('-pop', help='population size', type=int, default=20)
parser.add_argument('-sample', help='sample size', type=int, default=10)
parser.add_argument('-mutate_prob_split', help='split mutate probability', type=float, default=0.5)
parser.add_argument('-mutate_prob_pi', help='pi mutate probability', type=float, default=0.3)
parser.add_argument('-finetune_e', help='finetune turns during the GA', type=int, default=70)
parser.add_argument('-b', help='batch size', type=int, default=256)
parser.add_argument('-device', help='device index', type=str, choices=['0', '1'], default='0')
parser.add_argument('-resume', help='resume file path', type=str, default=None)
parser.add_argument('-r', help='resume epoch', type=int, default=0)
parser.add_argument('-point', help='resume file path to draw', type=str, default=None)
parser.add_argument('-dataset', help='which dataset to use', type=str, choices=['cifar10', 'imagenet'], default='cifar10')
parser.add_argument('-test', help='test or not', type=bool, default=False)
parser.add_argument('-acc', help='train many for acc', type=bool, default=False)

# def str2bool(str):
#     return True if str.lower() == 'True' else False

parser.add_argument('-predict', help='use acc module to predict or not', type=bool, default=False)
parser.add_argument('-greedy', help='use greedy or not', type=bool, default=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']= args.device
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if args.net == 'mobilenet':
    qmodel = qmobilenet(class_num=1000)
    qmodel = qmodel.cuda()
    block_type = [BasicConv2d, DepthSeperabelConv2d, QLinear]
    weight_path = conf.mobilenet_path
else:
    raise ValueError(f'the net {args.net} is not included')

ga = GeneticAlgorithm(
    arch=args.net,
    qmodel=qmodel,
    weight_path=weight_path,
    epoch=args.epoch, 
    pop_size=args.pop, 
    sample_size=args.sample, 
    mutate_prob_split=args.mutate_prob_split, 
    mutate_prob_pi=args.mutate_prob_pi,
    batch_size=args.b,
    block_type=block_type,
    finetune_e=args.finetune_e,
    resume_epoch=args.r,
    dataset=args.dataset,
    test=args.test,
    acc_module=args.acc,
    predict=args.predict,
    greedy = args.greedy
)

pop = {}
point = {}
if args.resume:
    load_file(pop, args.resume)
    load_point(point, args.point)
    print('恢复训练')
    print(pop)

choose = 'run'
if args.acc == True:
    choose = 'acc'
if args.test == True:
    choose = 'test'
if args.predict == True:
    choose = 'predict'

match choose:
    case 'run':
        best = ga.run(verbose=True, init_pop=pop, init_point=point)
    case 'test':
        # ga.test(verbose=True)
        # ga.relat()
        ga.predict_test()
    case 'acc':
        ga.acc_module()
    case 'predict':
        ga.predict_run()
