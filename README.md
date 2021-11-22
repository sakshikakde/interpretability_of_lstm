# References 
https://github.com/ayaabdelsalam91/Input-Cell-Attention       
https://github.com/pranoyr/cnn-lstm     

# dataset
https://www.csc.kth.se/cvap/actions/     
Use only running and walking dataset for now.
You need to run create dataset to extract first 70 frames, and the run generate.sh. Change the appropriate variables in generate.sh

# model
https://drive.google.com/file/d/1fLNivCLBySCBPyb_WkozNTvTzkre9NV3/view?usp=sharing
# How to run code 
## Train

python3 main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 50 --num_workers 0  --annotation_path ./data/kth/annotation/kth1.json --video_path ./data/kth/image_data/  --dataset kth --sample_size 150 --lr_rate 1e-4 --n_classes 2

## Test for video classification output 
 python3 inference.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>
    
## Get feature temporal Saliency
python3 feature_saliency.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>

## Get frame Saliency

    





