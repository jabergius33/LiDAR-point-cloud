#ADDED FOR BINARY EVAL OF NETWORK 
#(date: 19-05-2022, Johan Bergius)
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

from collections import Counter 
import collections
import numpy as np
import time
import os

# Local imports
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from scipy.spatial import distance

# input function
def read_kitty(bin_file_name:str=None, label_file_name:str=None):
    "opens a point cloud frame following kitty formatting"
    if label_file_name == None:
        assert False, 'labels does not exist'
    
    if label_file_name:
        with open(label_file_name, mode='rb') as file:
            label = np.fromfile(file, dtype=np.int16)
            label = np.reshape(label, (-1,2))

    
    return label


def write_kitty(folder_name:str, dataframes, labels=None, names=None): #NOT USED
    "writes numpy point cloud data as kitty format"
    names = names if names else [str(i) for i in range(len(dataframes))]
    if not os.path.isdir(folder_name):
        os.mkdir(os.path.join(folder_name))
        os.mkdir(os.path.join(folder_name, 'velodyne'))
        os.mkdir(os.path.join(folder_name, 'labels'))
        
    # make velodyne data
    vel_dir = os.path.join(folder_name, 'velodyne')
    for data, name in zip(dataframes, names):
        data.tofile(f'{vel_dir}/{name}.bin', sep='', format='%s')
    
    # make label data
    if labels:
        lab_dir = os.path.join(folder_name, 'labels')
        for data, name in zip(labels, names):
            data.tofile(f'{lab_dir}/{name}.label', sep='', format='%s')  




def main(args):
    prediction_path = args.prediction_folder
    dataset = args.dataset_folder 
    print()
    print("*"*40)
    print("Data directory:")
    print("Prediction_path: \n",prediction_path)
    print("Sequence_path: \n",dataset)
    print("*"*40)
    print()
    y_true = None
    y_pred = None
    
    sequences_to_eval =  ['30','34','35','36','76' ]
    for inp in sequences_to_eval:  

        Input_folder = os.path.join(dataset, inp)
        scan_path = os.path.join(Input_folder, 'velodyne')
        label_path = os.path.join(Input_folder, 'labels')

        pred_path = os.path.join(prediction_path, inp)
        pred_path = os.path.join(pred_path, 'predictions')

        print(f"Fetching data in SEQ: {inp}")
        
        pred_names = sorted([os.path.join(dp, f) 
                  for dp, dn, fn in os.walk(os.path.expanduser(pred_path)) 
                  for f in fn])

        label_names = sorted([os.path.join(dp, f) 
                   for dp, dn, fn in os.walk(os.path.expanduser(label_path)) 
                   for f in fn])

        pred_path_len = len(pred_path)+1
        label_path_len = len(label_path)+1
        assert all(pred[pred_path_len:-6] == label[label_path_len:-6] for pred, label in zip(pred_names, label_names))
        
        for label, pred in zip(label_names, pred_names):

        
            labels = read_kitty(None, label)    
            pred2 = read_kitty(None, pred) 
        
            #Sanity check
            assert labels.shape == pred2.shape, 'Error. Invalid data format!' 
            if y_true is None:
                y_true = labels.flatten() 
            else:   
                y_true = np.concatenate((y_true, labels.flatten()))

            if y_pred is None:
                    y_pred = pred2.flatten()
            else:
                y_pred = np.concatenate((y_pred, pred2.flatten()))


    print()             
    print("*"*40)
    print("EVAL SCORE:")
    print(f'Recall:    \t{recall_score(y_true, y_pred, average=None)[1]*100:.3f}%',)
    print(f'Precision: \t{precision_score(y_true, y_pred, average=None)[1]*100:.3f}%',)
    print(f'F1:        \t{f1_score(y_true, y_pred , average=None)[1]*100:.3f}%',)
    print("*"*40)     
    print('Finished script')
               
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prediction-folder', type=str, default='', help='Provide path to prediction label folder (sequence folder with the label files (SemanticKITTI format))',required=True)
    parser.add_argument('--dataset-folder', type=str, default='', help='dataset sequence path (SemanticKITTI format)', required=True)
        
        
    args = parser.parse_args()
    print("*"*40)
    print("Eval on WADS data (Seq: 30,34,35,36,37)")
    print("(Data must be in the official SemanticKITTI format!)")
    print("*"*40)
    #print(' '.join(sys.argv))
    main(args)
    
    

''' 
## GUIDE:
============================
INFO:
PREDICTION_PATH : 'path to prediction lablels ('sequence folder' according to the SemanticKITTI structure'
DATA_PATH : 'path to the dataset ('sequences folder' for all labels files (data must be in SemanticKITTI format ) )'

============================
Example path:
DATA_PATH       = './dataset/sequences'
PREDICTION_PATH = './predicted_labels/sequences'

============================
RUN (terminal):
python Result.py --prediction-folder [PREDICTION_PATH] --dataset-folder [DATA_PATH]
'''
#python Result.py --prediction-folder /home/stud/j/johanb17/Cylinder3D-master/sequences --dataset-folder /home/stud/j/johanb17/MasterThesisCode/sequences

