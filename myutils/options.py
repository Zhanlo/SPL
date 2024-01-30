import argparse
import sys
import yaml

from config import parser as _parser

parser = argparse.ArgumentParser(description='Lottery Jackpots Exsit in Pre-trained Model')

parser.add_argument('--model_name', type=str, default="safe_sub")
parser.add_argument('--dataset', default='CIFAR10')
parser.add_argument('--n_labels', type=int, default=2400)
parser.add_argument('--n_unlabels', type=int, default=20000)
parser.add_argument('--n_valid', type=int, default=5000)
parser.add_argument('--l_batch_size', type=int, default=512)
parser.add_argument('--ul_batch_size', type=int, default=512)
parser.add_argument('--test_batch_size', type=int, default=512)
parser.add_argument('--n_class', type=int, default=6)
parser.add_argument('--tot_class', type=int, default=10)
parser.add_argument('--ratio', type=float, default=0.6)
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--width', type=int, default=2)
parser.add_argument('--step_size_m', type=int, default=20)
parser.add_argument('--step_size_w', type=int, default=20)
parser.add_argument('--consis_coef', type=float, default=10.0)
parser.add_argument('--iteration', type=int, default=10)
parser.add_argument('--alg', default='PI')
parser.add_argument(
    '--lr_m',
    type=float,
    default=0.1,
    help='Learning rate for train. default:0.1'
)
parser.add_argument(
    '--lr_w',
    type=float,
    default=0.1,
    help='Learning rate for train. default:0.1'
)
parser.add_argument(
    '--epochs_m',
    type=int,
    default=30,
    help='The num of epochs to train. default:30'
)
parser.add_argument(
    '--epochs_w',
    type=int,
    default=30,
    help='The num of epochs to train. default:30'
)
parser.add_argument(    
    "--config", 
    help="Config file to use (see configs dir)", 
    default=None
)

parser.add_argument(
    '--use_dali',
    action='store_true',
    help='whether use dali module to load data'
)

parser.add_argument(
    "--label-smoothing",
    type=float,
    help="Label smoothing to use, default 0.0",
    default=0.0,
)

parser.add_argument(
    "--warmup_length", 
    default=0, 
    type=int, 
    help="Number of warmup iterations"
)

parser.add_argument(
	'--gpus',
	type=int,
	nargs='+',
	default=0,
	help='Select gpu_id to use. default:[0]',
)


parser.add_argument(
	'--data_set',
	type=str,
	default='cifar10',
	help='Select dataset to train. default:cifar10',
)

parser.add_argument(
	'--data_path',
	type=str,
	default='/home/userhome/datasets/cifar',
	help='The dictionary where the input is stored. default:',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored. default:./experiments'
)

parser.add_argument(
    '--resume',
    action='store_true',
    help='Load the model from the specified checkpoint.'
)

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='wideresnet',
    help='Architecture of model. default:vgg_cifar'
)



parser.add_argument(
    '--train_batch_size',
    type=int,
    default=50,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=50,
    help='Batch size for validation. default:256'
)
parser.add_argument(
    "--freeze_weights",
    action="store_true",
    help="Whether or not to train only subnet (this freezes weights)",
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)



parser.add_argument(
    "--optimizer", 
    help="Which optimizer to use", 
    default="sgd"
)

parser.add_argument(
    "--no_bn_decay", 
    action="store_true", 
    default=False, 
    help="No batchnorm decay"
)

parser.add_argument(
    "--lr_policy", 
    default="cos", 
    help="Policy for the learning rate."
)

parser.add_argument(
    "--lr_adjust", 
    default=30, 
    type=int, 
    help="Interval to drop lr"
)

parser.add_argument(
    "--lr_gamma", 
    default=0.1, 
    type=float, 
    help="Multistep multiplier"
)
parser.add_argument(
    "--prune_rate", 
    default=0.9, 
    type=float, 
    help="Prune rate"
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='The weight decay of loss. default:1e-4'
)

parser.add_argument(
    '--test_only',
    action='store_true',
    help='Test only?'
)

parser.add_argument(
    "--nesterov",
    default=False,
    action="store_true",
    help="Whether or not to use nesterov for SGD",
)

parser.add_argument(
    "--conv_type", 
    type=str, 
    default=None, 
    help="Conv type of conv layer. Default: PretrainConv. optional: DenseConv"
)

parser.add_argument(
    '--pruned_model',
    type=str,
    default=None,
    help='Path of the pruned model'
)

parser.add_argument(
    "--layerwise", 
    type=str, 
    default="l1", 
    help="Layerwise pruning rate. Default: l1. optional: uniform"
)

parser.add_argument(
    "--bn_type", 
    type=str, 
    default="LearnedBatchNorm", 
    help="BN type of conv layer. Optional: NonAffineBatchNorm"
)

parser.add_argument(
    '--debug',
    action='store_true',
    help='input to open debug state')


args = parser.parse_args()

override_args = _parser.argv_to_vars(sys.argv)


# load yaml file
yaml_txt = open(args.config).read()

# override args
loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
#print(loaded_yaml)
for v in override_args:
    loaded_yaml[v] = getattr(args, v)

print(f"==> Reading YAML config from {args.config}")
args.__dict__.update(loaded_yaml)

