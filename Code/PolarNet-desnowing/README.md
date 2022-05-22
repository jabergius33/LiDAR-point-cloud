# PolarNet for binary segmentation of snow

The code is tested on linux ubuntu 20.04 (conda version: 4.10.3).


#### Pre-setup:
##### Create a environment:
  * conda create -n myenv python=3.6
 ##### Activate it using:
  * conda activate myenv
##### Install packages:
  * pip install -r requirements.txt
##### Confirm that you are using torch 1.10.x+cu102
```
   python -c "import torch; print(torch.version)"
```
##### Install torch-scatter:
  * pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
##### Install additional package:
* pip install -U scikit-learn
* conda install tensorboard

  
  
### How to use it:


#### Training (train_SemanticKITTI.py)

* Run: train_SemanticKITTI.py  

##### Args:
*  --data_dir : path to data sequences folder

##### Example:
```
python train_SemanticKITTI.py --data_dir data/sequences/
```


#### Generate prediction labels (test_pretrain_SemanticKITTI.py)
1. Create a '/out' folder to store the output
2. Generate prediction by running: test_pretrain_SemanticKITTI.py

##### Args:
*  --data_dir : path to data sequences folder
* --model_save_path : path to saved model 

##### Example:
```
python test_pretrain_SemanticKITTI.py --data_dir data/sequences/ --model_save_path model.pt
```


#### Evaluate file (Result.py)    
Inputs:
* --prediction-folder : Path to predictions folder (/pred/sequences)
* --dataset-folder : Path to the dataset (/dataset/sequences)   
 ##### Example:
 ```
  python Result.py --prediction-folder /pred/sequences --dataset-folder /dataset/sequences
 ``` 
  
  ### Acknowledgment
Acknowledgment are given to [PolarNet](https://github.com/edwardzhou130/PolarSeg/blob/master/README.md)
