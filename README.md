# References 
https://github.com/ayaabdelsalam91/Input-Cell-Attention       
https://github.com/pranoyr/cnn-lstm     

# Dataset
## Create your data
``` mkdir data ```   
``` cd data```      
``` mkdir kth ```   
``` mkdir raw_dataset```
``` cd raw_dataset ```    
``` wget http://www.nada.kth.se/cvap/actions/walking.zip ```    
``` wget http://www.nada.kth.se/cvap/actions/running.zip```


``` cd base_directory```    
Trim videos:  ```python3 ./utils/trim_videos.py  ```     
Generate data: ```./utils/generate_data.sh ```
  


## Download pre processed data
```wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sMn_BGhqmGdgZ0JKlodBwVB0Z5ppNqi_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sMn_BGhqmGdgZ0JKlodBwVB0Z5ppNqi_" -O data.zip && rm -rf /tmp/cookies.txt```       
```unzip data.zip```






# Models
## RESNET101     

https://drive.google.com/file/d/1Ft0dUgL7im65hQev-BYElXnzpud2Xyv0/view?usp=sharing# 

## RESNET18
### 70 frames
https://drive.google.com/file/d/10H_mJnJABIZZnIDSEUdvabm4gtk3JJMX/view?usp=sharing

### 30 frames
https://drive.google.com/file/d/1BVMcI_Bd-vtJNN0hO5gN1vI18M5NQ8Yx/view?usp=sharing

# How to run code 
## Train

```python3 main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 50 --num_workers 0  --annotation_path ./data/kth/annotation/kth1.json --video_path ./data/kth/image_data/  --dataset kth --sample_size 150 --lr_rate 1e-4 --n_classes 2```

## Test for video classification output 
 ```python3 inference.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>```
    
## Get feature temporal Saliency -  need updates
python3 feature_saliency.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>

## Get frame Saliency
If you want to use TSR, set a flaf to true in frame_saliency.py      
```python3 frame_saliency.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path /home/sakshi/courses/CMSC828W/cnn-lstm/snapshots/cnnlstm/cnnlstm-Epoch-72-Loss-0.8406717499097188_Nov-23-2021_0311.p```
    
<!-- ## new model
https://drive.google.com/file/d/11KT6b9pKAwP7zUBMyrJDlZRsG8qsTD5f/view?usp=sharingS -->






