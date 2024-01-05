
import warnings
import shutil

warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json

os.chdir(os.path.dirname(os.path.realpath(__file__)))

import matplotlib

matplotlib.use('agg')
import clustering as CLS
import datasets as data

import netlib as netlib
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='cars196', type=str, help='Dataset to use.')#cars196

parser.add_argument('--lr', default=0.00001, type=float,
                    help='Learning Rate for network parameters.')

parser.add_argument('--fc_lr_mul', default=0, type=float,
                    help='OPTIONAL: Multiply the embedding layer learning rate by this value. '
                         'If set to 0, the embedding layer shares the same learning rate.')

parser.add_argument('--n_epochs', default=150, type=int, help='Number of training epochs.')

parser.add_argument('--kernels', default=8, type=int,
                    help='Number of workers for pytorch dataloader.')

parser.add_argument('--bs', default=112, type=int, help='Mini-Batchsize to use.')

parser.add_argument('--samples_per_class', default=8, type=int,
                    help='Number of samples in one class drawn before choosing the next class. '
                         'Set to >1 for losses other than ProxyNCA.')

parser.add_argument('--seed', default=23, type=int, help='Random seed for reproducibility.')

parser.add_argument('--scheduler', default='step', type=str,
                    help='Type of learning rate scheduling. Currently: step & exp.')

# #### Network parameters
parser.add_argument('--embed_dim', default=256, type=int,
                    help='Embedding dimensionality of the network. Note: '
                         'in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.')

parser.add_argument('--arch', default='attention', type=str,
                    help='Network backend choice: resnet50, googlenet.')

parser.add_argument('--not_pretrained', action='store_true',
                 help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')

# #### Setup Parameters
parser.add_argument('--gpu', default=0, type=int, help='GPU-id for GPU to use.')

# ## Paths to datasets and storage folder
parser.add_argument('--source_path', default=os.getcwd() + '/Datasets', type=str,
                    help='Path to training data.')

# #### Read in parameters
opt = parser.parse_args()

"""============================================================================"""
opt.source_path += '/' + opt.dataset
opt.save_path += '/' + opt.dataset

if opt.dataset == 'online_products':
    opt.k_vals = [1, 10, 100, 1000]
if opt.dataset == 'in-shop':
    opt.k_vals = [1, 10, 20, 30, 50]
if opt.dataset == 'vehicle_id':
    opt.k_vals = [1, 5]

if opt.loss == 'proxynca':
    opt.samples_per_class = 1
else:
    assert not opt.bs % opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

if opt.loss == 'npair' or opt.loss == 'proxynca': opt.sampling = 'None'

opt.pretrained = not opt.not_pretrained

"""============================================================================"""
# ################## GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


# """============================================================================"""
# ################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic = True
np.random.seed(opt.seed);
random.seed(opt.seed)
torch.manual_seed(opt.seed);
torch.cuda.manual_seed(opt.seed);
torch.cuda.manual_seed_all(opt.seed)
print("*" * 119)

"""==================================NETWORK SETUP=========================================="""
# #################### NETWORK SETUP ##################
opt.device = torch.device('cuda')
# Depending on the choice opt.arch, networkselect() returns the respective network model
model = netlib.networkselect(opt)
model_path="/mnt/disk2/hamideh.rafiee/sampling_/Training_Results/" \
           "cars196/resnet_cub_triplet_semi_4/checkpoint.pth.tar"
checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint['state_dict'])
_ = model.to(opt.device)

print("model loaded from checkpoint ...")

"""============================================================================"""


dataloaders = data.give_dataloaders(opt.dataset, opt, model)
print("data loaded .....")
print(dataloaders.keys())
print(dataloaders["testing"].dataset.image_dict.keys())#training

cluster_dict = CLS.clustering(dataloaders["testing"].dataset.image_dict, model, opt)

# ------------------------------------------------------
destination_path="/mnt/disk2/hamideh.rafiee/sampling_/Training_Results/CLUSTER/"
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

for key in cluster_dict.keys():
    if not os.path.exists(destination_path+str(key)):
        os.makedirs(destination_path+str(key))
    for i in cluster_dict[key]:
        name=i.split("/")[-1]
        shutil.copy(i, destination_path+str(key)+"/"+name)










