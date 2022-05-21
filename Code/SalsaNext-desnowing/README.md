# SalsaNext for binary segmentation of snow
<img src="https://github.com/jabergius33/LiDAR-point-cloud/blob/main/gifs/With_%20FalsePositive/salsanext.gif" alt="drawing" width="400"/>

The code is implmeted on linux ubuntu 20.04 (conda version 4.10.3).


#### Pre-config:
##### Create a environment with:
  ``` 
  conda env create -f salsanext_cuda10.yml --name salsanext
  ```
 ##### Activate it using:
  ```
  conda activate salsanext
  ```
##### Install additional package:
  * pip install future
  * conda install tensorboard
  * pip install -U scikit-learn
#### Dataset:
* Make sure your dataset is stored in a /sequence folder according to the SemanticKITTI doc. (Dataset path is the path to this folder)
 
### How to use it for binary segmentation:
1. Update salsanext.yml and tasks/semantic/config/labels/semantic-kitti.yaml (for your dataset)
2. Train the model using: train.sh  (Train the network. config found in salsanext.yml)
3. Generate prediction labels using: eval.sh (Generate prediction labels)
4. Run: Result.py to evaluate the predicted output (Evaluate the prediction on Recall, Precision & F1)


#### Training file (train.sh) (potentially need to chmod +x the file)
  Inputs:
* -d (string): Dataset path (path to /sequence folder according to SemanticKITTI doc.)
* -a (string): network config (./salsanext.yml)
* -m (string): model name (default: salsanext)    
* -l (string): Path to store log/model
* -c (string): Gpu-ID (to use)
##### Example:
```
./train.sh -d /data_path -a /salsanext.yml -m salsanext -l /logfile -c 0
```

#### Generate prediction labels (eval.sh) (potentially need to chmod +x the file)
  Inputs:
* -d (string): Dataset path (path to /sequence folder according to SemanticKITTI doc.)
* -p (string): Path to store/save predictions 
* -m (string): Folder to pre-trained model      
* -s (string): Data split to infer (test/train/valid) 
* -c (string): MC sampling (default: 30) 
 ##### Example:
```
./eval.sh -d /data_path -p ./pred -m /model/logs/2022-3-08-09:05 -s test -n salsanext -c 30 
```

#### Evaluate file (Result.py)    
Inputs:
* --prediction-folder : Path to predictions folder
* --dataset-folder : Path to the dataset (/data_path/sequences)   
 ##### Example:
 ```
 python Result.py --prediction-folder /pred --dataset-folder /data_path/sequences
 ``` 
  
### Acknowledgment
Acknowledgment are given to the opensource code: [salsanext](https://github.com/Halmstad-University/SalsaNext/blob/master/README.md)

