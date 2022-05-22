#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#ORIGINAL FILE IS ALTERED FOR BINARY CLASSIFCATION 
#(date: 19-05-2022, Johan Bergius)

import os
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from sklearn.metrics import f1_score

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,collate_fn_BEV_test,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset

import warnings
warnings.filterwarnings("ignore")


def main(args):
    data_path = args.data_dir
    test_batch_size = args.test_batch_size
    model_save_path = args.model_save_path
    output_path = args.test_output_path
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

    # prepare miou 
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[:]
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label]

    

    # prepare model
    my_BEV_model=BEV_Unet(n=19, n_height = 32, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding)
    

    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    
 
    if os.path.exists(model_save_path):
        my_model.load_state_dict(torch.load(model_save_path)) 
        print()
        print("Using existing model!")
    else:
        print("No existing model!")
    
    my_model.to(pytorch_device)

    # prepare dataset  

    #print("data_path ",data_path)    
    test_pt_dataset = SemKITTI(data_path , imageset = 'test', return_ref = True)
    val_pt_dataset = SemKITTI(data_path , imageset = 'val', return_ref = True)
    if model == 'polar':
        test_dataset=spherical_dataset(test_pt_dataset,grid_size= grid_size, ignore_label = 2, fixed_volume_space = True, return_test= True)
        val_dataset=spherical_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 2, fixed_volume_space = True)
    
   
    test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV_test,
                                                    shuffle = False,
                                                    num_workers = 4)
    
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)
    
    # validation
    print('*'*80)
    print('Test network performance on validation split')
    print('*'*80)
    pbar = tqdm(total=len(val_dataset_loader))
    my_model.eval()
    hist_list = []
    
    vali_pred = []
    vali_Y =  []
    
    with torch.no_grad():
        for i_iter_val,(_,val_vox_label,val_grid,val_pt_labs,val_pt_fea) in enumerate(val_dataset_loader):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
            val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)


            predict_labels = my_model(val_pt_fea_ten, val_grid_ten)
            predict_labels =torch.round(predict_labels[:,0,:,:,:])
            
            predict_labels = predict_labels.cpu().detach().numpy()
            for count,i_val_grid in enumerate(val_grid):
                if len(vali_pred) ==0:
                    vali_pred = predict_labels[count,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]].flatten()
                    vali_Y = val_pt_labs[count][:,0].flatten()
                else:
                    vali_pred  = np.concatenate((vali_pred, predict_labels[count,val_grid[count][:,0],val_grid[count][:,1],val_grid[count][:,2]].flatten()))
                    vali_Y = np.concatenate((vali_Y, val_pt_labs[count][:,0].flatten()))

                
            pbar.update(1)

    
    del val_vox_label,val_grid,val_pt_fea,val_grid_ten
    pbar.close()
    print()
    print('F1 for validtion: %.3f' % (f1_score(vali_Y, vali_pred , average=None)[1]*100))

    
    # test
    print('*'*80)
    print('Generate predictions for test split')
    print('*'*80)
    pbar = tqdm(total=len(test_dataset_loader))
    with torch.no_grad():
        for i_iter_test,(_,_,test_grid,_,test_pt_fea,test_index) in enumerate(test_dataset_loader):
            # predict
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in test_grid]

            predict_labels = my_model(test_pt_fea_ten,test_grid_ten) #PREICTION
            
            predict_labels =torch.round(predict_labels[:,0,:,:,:]).type(torch.uint8) 
            predict_labels = predict_labels.cpu().detach().numpy()
            # write to label file
            for count,i_test_grid in enumerate(test_grid):
                test_pred_label = predict_labels[count,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]]

                test_pred_label = np.expand_dims(test_pred_label,axis=1)
                save_dir = test_pt_dataset.im_idx[test_index[count]]
                _,dir2 = save_dir.split('/sequences/',1)
                
                new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'        

                
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                test_pred_label = test_pred_label.astype(np.uint32)


                
                test_pred_label.tofile(new_save_dir)
            pbar.update(1)
    del test_grid,test_pt_fea,test_index
    pbar.close()


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='data/sequences/') 
    parser.add_argument('-p', '--model_save_path', default='Model1.pt')
    parser.add_argument('-o', '--test_output_path', default='/out')
    parser.add_argument('-m', '--model', choices=['polar','traditional'], default='polar')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32], help='grid size (default: [480,360,32])')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size (default: 1)')
    
    args = parser.parse_args()
    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')

    print(' '.join(sys.argv))
    print(args)
    main(args)
