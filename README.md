# References 
https://github.com/ayaabdelsalam91/Input-Cell-Attention       
https://github.com/pranoyr/cnn-lstm     

# dataset
## Create your data
mkdir data    
cd data
mkdir kth    
mkdir raw_dataset
cd raw_dataset     
wget http://www.nada.kth.se/cvap/actions/walking.zip     
wget http://www.nada.kth.se/cvap/actions/running.zip


cd Home    
Trim videos:  python3 ./utils/trim_videos.py       
Generate data: ./utils/generate_data.sh 
  


## Download pre processed data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sMn_BGhqmGdgZ0JKlodBwVB0Z5ppNqi_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sMn_BGhqmGdgZ0JKlodBwVB0Z5ppNqi_" -O data.zip && rm -rf /tmp/cookies.txt       
unzip data.zip




https://www.csc.kth.se/cvap/actions/     
Use only running and walking dataset for now.
You need to run create dataset to extract first 70 frames, and the run generate.sh. Change the appropriate variables in generate.sh     
the file structure should be like this:
![alt text](https://github.com/sakshikakde/interpretability_of_lstm/blob/kth/doc_images/Screenshot%20from%202021-11-22%2016-48-26.png)


raw videos is the actual kth dataset,
trimmed one is after trim_videos.py


# model
## RESNET101     

https://drive.google.com/file/d/1Ft0dUgL7im65hQev-BYElXnzpud2Xyv0/view?usp=sharing# 

## RESNET18
https://drive.google.com/file/d/10H_mJnJABIZZnIDSEUdvabm4gtk3JJMX/view?usp=sharing

How to run code 
## Train

python3 main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 50 --num_workers 0  --annotation_path ./data/kth/annotation/kth1.json --video_path ./data/kth/image_data/  --dataset kth --sample_size 150 --lr_rate 1e-4 --n_classes 2

## Test for video classification output 
 python3 inference.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>
    
## Get feature temporal Saliency
python3 feature_saliency.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>

## Get frame Saliency
## TSR
python3 frame_saliency.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path /home/sakshi/courses/CMSC828W/cnn-lstm/snapshots/cnnlstm/cnnlstm-Epoch-72-Loss-0.8406717499097188_Nov-23-2021_0311.p
    
<!-- ## new model
https://drive.google.com/file/d/11KT6b9pKAwP7zUBMyrJDlZRsG8qsTD5f/view?usp=sharingS -->






