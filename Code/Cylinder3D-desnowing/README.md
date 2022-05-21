# Cylinder for binary segmentation of snow

The code is implmeted on linux ubuntu 20.04 (conda version: 4.10.3).


#### Pre-config:
##### Create a environment with:
  * conda env create -f environment.yml
 ##### Activate it using:
  * conda activate Cylinder
##### Install additional package:
  * pip install -U scikit-learn
  
### How to use it:


#### Training (potentially need to chmod +x the file)

1. Modify the config file at 'config/semantickitti.yaml' (Setup and dataset path)
2. Train the network by running: 
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


#### Evaluate file (Result.py)    
Inputs:
* --prediction-folder : Path to predictions folder (/pred)
* --dataset-folder : Path to the dataset (/dataset/sequences/)   
 ##### Example:
 ```
 python Result.py --prediction-folder /pred --dataset-folder /dataset/sequences/
 ``` 
  
  ### Acknowledgment
Acknowledgment are given to the open source code of 