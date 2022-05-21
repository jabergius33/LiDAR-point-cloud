# -*- coding:utf-8 -*-
# author: Ptzu
# @file: demo_folder.py


#ORIGINAL FILE IS ALTERED FOR BINARY CLASSIFCATION 
#2022 Johan Bergius.
import time
import os
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml
import torch.nn as nn 

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")


def build_dataset(dataset_config,
                  data_dir,
                  grid_size=[480, 360, 32],
                  demo_label_dir=None):

    if demo_label_dir == '':
        imageset = "demo"
    else:
        imageset = "val"
    label_mapping = dataset_config["label_mapping"]

    SemKITTI_demo = get_pc_model_class('SemKITTI_demo')

    demo_pt_dataset = SemKITTI_demo(data_dir, imageset=imageset,
                              return_ref=True, label_mapping=label_mapping, demo_label_path=demo_label_dir)

    demo_dataset = get_model_class(dataset_config['dataset_type'])(
        demo_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )
    demo_dataset_loader = torch.utils.data.DataLoader(dataset=demo_dataset,
                                                     batch_size=1,
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=False,
                                                     num_workers=4)

    return demo_dataset_loader

def main(args):
    #Create SEQ folder
    if not os.path.isdir(os.path.join(f'./sequences/')):
        os.mkdir(os.path.join(f'./sequences/'))
        
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    data_dir = args.demo_folder
    demo_label_dir = args.demo_label_folder
    
    
    demo_batch_size = 1
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
     

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[:] - 1 
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
    

    
    #************ Select Model ************
    model_load_path = ''
    #model_load_path = './SavedModel/Model1.pt'
    #**************************************
    
    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        print()
        print(f"Using model: {model_load_path}")
        my_model = load_checkpoint(model_load_path, my_model)
    else:
        #assert False, 'No model to evaluate, please select one!'
        print("*"*20)
        print("WARNING")
        print('No model is selected!')

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax, BCEloss2 = loss_builder.build(wce=True, lovasz=True, num_class=num_class, ignore_label=ignore_label)
    loss_criterion = nn.BCELoss() 

    
    ##For all sequences
    sequences_to_eval =  ['30','34','35','36','76' ]
    for inp in sequences_to_eval:  
        print()
        print(f"Sequence: {inp}")
        data_dir_tmp = os.path.join(data_dir, f'{inp}', 'velodyne')
        #Create SEQ folder
        if not os.path.isdir(os.path.join(f'./sequences/{inp}')):
            os.mkdir(os.path.join(f'./sequences/{inp}'))
    

        demo_dataset_loader = build_dataset(dataset_config, data_dir_tmp, grid_size=grid_size, demo_label_dir=demo_label_dir)
        with open(dataset_config["label_mapping"], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        inv_learning_map = semkittiyaml['learning_map_inv']

        my_model.eval()
        hist_list = []
        demo_loss_list = []
        with torch.no_grad():
            for i_iter_demo, (_, demo_vox_label, demo_grid, demo_pt_labs, demo_pt_fea) in enumerate(demo_dataset_loader):
                demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in demo_pt_fea]
                demo_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in demo_grid]
                demo_label_tensor = demo_vox_label.type(torch.LongTensor).to(pytorch_device)


                predict_labels = my_model(demo_pt_fea_ten, demo_grid_ten, demo_batch_size)

                predict_labels.detach()            
                loss = BCEloss2(predict_labels[:,0,:,:,:], demo_label_tensor.float()) 


                predict_labels = torch.round(predict_labels[:,0,:,:,:]) #predictions!
                predict_labels = predict_labels.cpu().detach().numpy()


                for count, i_demo_grid in enumerate(demo_grid):
                    hist_list.append(fast_hist_crop(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]].astype(int), demo_pt_labs[count].astype(int),unique_label))


                    inv_labels = np.vectorize(inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]) 

                    
                    inv_labels = inv_labels.astype('uint32')
                    
                    
                    ##create label folder
                    if not os.path.isdir(os.path.join(f'./sequences/{inp}', 'predictions')):
                        os.mkdir(os.path.join(f'./sequences/{inp}', 'predictions'))
                        

                    ##Fetch all label names!
                    filenames = sorted(next(os.walk(f'{data_dir_tmp[:-8]}/labels'))[2])  

                    #Predictions to file
                    outputPath = f'./sequences/{inp}/predictions/{filenames[i_iter_demo]}' 
                    inv_labels.tofile(outputPath)                
                    print("saving: " + outputPath)

    print("Done with predictions!")     

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('--demo-folder', type=str, default='', help='Path to the folder containing test lidar scans', required=True)
    parser.add_argument('--demo-label-folder', type=str, default='', help='not used atm!')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
    
    
##Generating a file to use for evaluation
