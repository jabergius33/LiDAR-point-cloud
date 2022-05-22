# Cylinder3D for binary segmentation of snow

The code is implmeted on linux ubuntu 20.04 (conda version: 4.10.3).


#### Pre-config:
##### Create a environment with:
  * conda env create -f environment.yml
 ##### Activate it using:
  * conda activate Cylinder
##### Install additional package:
  * pip install -U scikit-learn
  
#### Dataset:
* Make sure your dataset is stored in a /sequence folder according to the SemanticKITTI doc. (Dataset path is the path to this folder)
 
  
### How to use it:
1. Modify semantickitti.yaml
2. Train
3. Modify demo_test_folder.py
4. Generate predictions (run demo_test_folder.py )
5. Eval (run Result.py)

#### Training (potentially need to chmod +x the file)

1. Modify the config file at 'config/semantickitti.yaml' (Setup and dataset path)
2. Create a folder 'logs_dir'
3. Train the network by running: 
```
sh train.sh
```

#### Generate prediction labels (demo_test_folder.py )
1. Update 'model_load_path' in "demo_test_folder.py" (model to infer)
3. Run: demo_test_folder.py

  Args:
* --demo-folder : Path to the dataset (sequences folder)
 ##### Example:
```
python demo_test_folder.py --demo-folder /dataset/sequences/
```


#### Evaluate (Result.py)    
 Args:
* --prediction-folder : Path to predictions folder (/pred/sequences)
* --dataset-folder : Path to the dataset (/dataset/sequences)   
 ##### Example:

 ```
  python Result.py --prediction-folder /pred/sequences --dataset-folder /dataset/sequences
 ``` 
  
  ### Acknowledgment
Acknowledgment is given to [Cylinder3d](https://github.com/xinge008/Cylinder3D/blob/master/README.md)
