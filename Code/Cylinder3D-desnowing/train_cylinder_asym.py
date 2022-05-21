# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

#ORIGINAL FILE IS ALTERED FOR BINARY CLASSIFCATION 
#2022 Johan Bergius.
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn 

from torchsummary import summary

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data


from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")

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

def main(args):
    #writer = SummaryWriter("tb")

    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[:] - 1 
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    
    #continue training model
    #model_load_path = '-'
    my_model = model_builder.build(model_config)
    #if os.path.exists(model_load_path):
        #my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"], amsgrad=True)
     
        
    loss_func, lovasz_softmax, BCEloss2 = loss_builder.build(wce=True, lovasz=True, num_class=num_class, ignore_label=ignore_label)
    loss_criterion = nn.BCELoss() 
    
    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0 
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10) 

        for i_iter, (_, train_vox_label, train_grid, tmp, train_pt_fea) in enumerate(train_dataset_loader):
               
            #training
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            train_batch_size = train_vox_label.shape[0] 
            

            ## forward + backward + optimize 
            outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            
            y = point_label_tensor.float()    
            outputs, y = flatten_binary_scores(outputs[:,0,:,:,:], y, ignore=2) 
            loss = loss_criterion(outputs, y ) 
    
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            
            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')            
            
        pbar.close()
        epoch += 1
        #writer.add_scalar('Loss',np.mean(loss_list), epoch) #tensorbaord 
        if epoch > 80 :  
            print("Saveing model")
            torch.save(my_model.state_dict(), "./MODEL1.pt") #save the model
        
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    #print(args)
    main(args)
    