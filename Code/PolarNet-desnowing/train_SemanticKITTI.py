#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#ORIGINAL FILE IS ALTERED FOR BINARY CLASSIFCATION 
#(date: 18-05-2022, Johan Bergius)

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter 
from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset


import warnings
warnings.filterwarnings("ignore")


from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    #print('initialize network with %s' % init_type)
    net.apply(init_func)  


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids) 
    init_weights(net, init_type, init_gain=init_gain)
    return net


def flatten_binary_scores(scores, labels, ignore=2):  
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels
    

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count=np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+1)
    hist=hist[unique_label,:]
    hist=hist[:,unique_label]
    return hist

def main(args):
    
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    check_iter = args.check_iter
    model_save_path = args.model_save_path
    compression_model = args.grid_size[2]
    grid_size = args.grid_size
    pytorch_device = torch.device('cuda:0')
    model = args.model
    if model == 'polar':
        fea_dim = 9
        circular_padding = True
    elif model == 'traditional':
        fea_dim = 7
        circular_padding = False

    #prepare miou
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[:]
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label]

   
    #prepare model
    my_BEV_model=BEV_Unet(n=19, n_height = 32, input_batch_norm = True, dropout = 0.2, circular_padding = circular_padding)
    
    init_net(my_BEV_model, init_type='kaiming', init_gain=0.02, gpu_ids=[])
    
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    
        
    my_model.to(pytorch_device)

    optimizer = optim.Adam(my_model.parameters(), lr=0.0005 , amsgrad=True)
    loss_func = torch.nn.BCELoss()

    

    
    # dataset
    print("data_path",data_path) #datapath/sequences/
    train_pt_dataset = SemKITTI(data_path, imageset = 'train', return_ref = True) 
    val_pt_dataset = SemKITTI(data_path, imageset = 'val', return_ref = True) 
    
    if model == 'polar':
        print("Polar setting")
        train_dataset=spherical_dataset(train_pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 2,
rotate_aug = True, fixed_volume_space = True)
        
        val_dataset=spherical_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 2, fixed_volume_space = True)
    

    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = train_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers = 4)
    
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    # training
    MAX_EPOCH=200
    
    epoch = 0
    my_model.train()
    global_iter = 0
    #writer = SummaryWriter("tb")
    
    while epoch < MAX_EPOCH:
        loss_list=[]
        
        pbar = tqdm(total=len(train_dataset_loader))
        for i_iter,(_,train_vox_label,train_grid,pt_labs,train_pt_fea) in enumerate(train_dataset_loader):
            
            # validation
            if (global_iter % 2600 == 0) & (epoch > 0) :
                my_model.eval()
                hist_list = []
                val_loss_list = []
                vali_pred = []
                vali_Y =  []
                with torch.no_grad():
                    for i_iter_val,(_,val_vox_label,val_grid,val_pt_labs,val_pt_fea) in enumerate(val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
                        val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten)
                    
                        output, y = flatten_binary_scores(predict_labels[:,0,:,:,:], val_label_tensor, ignore=2)               
                        loss = loss_func(output, y.float())
                        
                        predict_labels =torch.round(predict_labels[:,0,:,:,:])                       
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count,i_val_grid in enumerate(val_grid):
                            if len(vali_pred) ==0:
                                vali_pred = predict_labels[count,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]].flatten().astype(int)
                                vali_Y = val_pt_labs[count][:,0].flatten()

                                
                            else:
                                vali_pred  = np.concatenate((vali_pred, predict_labels[count,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]].flatten().astype(int)))
                                vali_Y = np.concatenate((vali_Y, val_pt_labs[count][:,0].flatten()))                                

                               
                            
                        val_loss_list.append(loss.detach().cpu().numpy())
                my_model.train()
                
                del val_vox_label,val_grid,val_pt_fea,val_grid_ten
                
                print('F1 %.3f' % (f1_score(vali_Y, vali_pred , average=None)[1]*100))
                print('Current val loss %.3f' % (np.mean(val_loss_list)))
               
           
            # training        
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor=train_vox_label.type(torch.LongTensor).to(pytorch_device) 


            
            # forward + backward + optimize
            predict_labels = my_model(train_pt_fea_ten,train_grid_ten) 
            outputs, y = flatten_binary_scores(predict_labels[:,0,:,:,:], point_label_tensor, ignore=2) 
            y = y.float()
            loss = loss_func(outputs, y)
            
            
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
      
            
            if global_iter % 1000 == 0:
                print('epoch %d iter %5d, loss: %.3f\n' % (epoch, i_iter, np.mean(loss_list)))

            # zero the parameter gradients
            optimizer.zero_grad()
            pbar.update(1) 
            global_iter += 1
        pbar.close()
        epoch += 1
        
        #writer.add_scalar('Loss',np.mean(loss_list), epoch) 
        if (epoch > 20) & (epoch % 20 == 0):
            print("Saving model")
            torch.save(my_model.state_dict(), model_save_path)
            

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='data/sequences/') 
    parser.add_argument('-p', '--model_save_path', default='./Model1.pt')
    parser.add_argument('-m', '--model', choices=['polar','traditional'], default='polar', help='training model (default: polar)')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32], help='grid size (default: [480,360,32])')
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type=int, default=1, help='batch size for validation (default: 1)')
    parser.add_argument('--check_iter', type=int, default=1000, help='validation interval (not used)')
    
    args = parser.parse_args()
    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')

    print(' '.join(sys.argv))
    #print(args)
    main(args)