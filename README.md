# References 
https://github.com/ayaabdelsalam91/Input-Cell-Attention       
https://github.com/pranoyr/cnn-lstm      

# How to run code 
## Train

python3 main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 50 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes 3

## Test for video classification output 
python3 inference.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes 3 --resume_path <model path>
    
## Get Saliency
python3 saliency.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes 3 --resume_path <model path>
    
## Plot Saliency 
python3 plotSaliency.py




