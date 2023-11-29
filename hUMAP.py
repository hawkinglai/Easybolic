from collections import OrderedDict
import numpy as np
import numba
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap

import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from models.pointmlp import Hype_pointMLP, Hype_pointMLPtest
# from models.Phpointmlp import Hype_pointMLP
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
from hutil import hype_triplet_losses, get_children_np
import numpy as np
import geoopt
import random

sns.set(style='white', rc={'figure.figsize':(10,10)})

import torch.backends.cudnn as cudnn
def load_pretrained(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # print(checkpoint)
    net = Hype_pointMLP()
    # print(net)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    net = net.to(device)
    if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = 'module.' + k  # add `module.`
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    return net

def validate(net, testloader):
    net.eval()
    device = 'cuda:0'
    # test_loss = 0
    test_true = []
    test_pred = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            
            _, data, n_points = get_children_np(data,kmin=600, kmax=1024)
            mar, pos_data, _ = get_children_np(data,starting=n_points,kmin=200, kmax=n_points//2)
            
            # _, logits_po = net(pos_data)
            # _, logits = net(data)


            # logits_po, _ = net(pos_data)
            logits, _ = net(data)

            test_true.append(label.cpu().numpy())
            # test_pred.append(logits_po.detach().cpu().numpy())
            test_pred.append(logits.detach().cpu().numpy())
            
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return test_true, test_pred


def validate2(net, testloader):
    net.eval()
    device = 'cuda:0'
    # test_loss = 0
    test_true = []
    test_pred = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            
            # _, data, n_points = get_children_np(data,kmin=600, kmax=1024)
            _, logits = net(data)


            # logits_po, _ = net(pos_data)
            # logits, _ = net(data)

            test_true.append(label.cpu().numpy())
            test_pred.append(logits.detach().cpu().numpy())
            # test_pred.append(logits_po.detach().cpu().numpy())
            
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return test_true, test_pred



if __name__ == '__main__':
    # this is raw humap, so you need to fine-tune

    mpath = '/home/kitahara/test/hyperbolic/EasyBolic/checkpoints/HypePointMLP-toyproof-Offv_pointmlp_hycore/best_checkpoint.pth'
    net = load_pretrained(mpath)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024),
                             num_workers=4, batch_size=32, shuffle=True, drop_last=False)
    # _, test_pred = validate(net,test_loader)
    _, test_pred = validate2(net,test_loader)

    hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid',
                                #   n_neighbors=10,
                                     random_state=13
                                     ).fit(test_pred)
    
    # hyperbolic_mapper2 = umap.UMAP(output_metric='hyperboloid',
    #                               n_neighbors=10,
    #                                  random_state=13
    #                                  ).fit(test_pred2)

    

    x = hyperbolic_mapper.embedding_[:, 0]
    y = hyperbolic_mapper.embedding_[:, 1]
    z = np.sqrt(1 + np.sum(hyperbolic_mapper.embedding_**2, axis=1))
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)

    # x2 = hyperbolic_mapper2.embedding_[:, 0]
    # y2 = hyperbolic_mapper2.embedding_[:, 1]
    # z2 = np.sqrt(1 + np.sum(hyperbolic_mapper2.embedding_**2, axis=1))
    # disk_x2 = x2 / (1 + z2)
    # disk_y2 = y2 / (1 + z2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(disk_x, disk_y, c=np.arange(test_pred.shape[0]), cmap='Spectral')
    boundary = plt.Circle((0,0), 1, fc='none', ec='k')
    ax.add_artist(boundary)
    ax.axis('off');

    # fig, ax = plt.subplots()
    # ax.scatter(disk_x, disk_y, c=np.arange(test_pred.shape[0]), cmap='Spectral')
    # ax.scatter(disk_x2, disk_y2, c=np.arange(test_pred2.shape[0]), cmap='Spectral')
    # boundary = plt.Circle((0,0), 1, fc='none', ec='k')
    # ax.add_artist(boundary)
    # ax.axis('off')

    plt.savefig('my_plot.png')
