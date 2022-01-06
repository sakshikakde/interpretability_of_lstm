# References 
https://github.com/ayaabdelsalam91/Input-Cell-Attention       
https://github.com/pranoyr/cnn-lstm     
# Introduction
As deep learning models become more commonplace, there is a rising need to accurately interpret
them. One way to do this involves the usage of different types of saliency methods. While saliency
has been well explored on single sample data in vision and language, the applicability of these meth-
ods in analyzing sequential or time-series data
remains relatively unexplored. In our work, we
look at tackling this problem of interpretability of
sequential data, employing saliency approaches.
In particular, we focus on interpreting real world
videos, which poses its own set of challenges. Our
results show a heavy computation load, demon-
strating the need for more efficient algorithms for
computing sequential saliency.

# Problem Definition and Setup
We formulate our approach as a binary classification prob-
lem and choose the two classes ‘walking’ and ‘running’
from the KTH action recognition Dataset (Schuldt et al.,
2004). All classes in the KTH dataset have the same individ-
uals performing different actions in a similar background.
We argue that this leads to the classifier relying more heavily
on causal features to make accurate decisions. In contrast choosing two classes such as ’playing badminton’ vs ’play-
ing the piano’ could result in the classifier relying on spuri-
ous features (background). ’Walking and ’running’ in partic-
ular have high visual similarity but have different temporal
characteristics. Combining these arguments we hypothesize
that by choosing these two specific classes the saliency maps
will closely resemble the optical flow. As optical flow can
be easily calculated this provides a good baseline for the
qualitative evaluation of the generated saliency maps. 



# How to run code 
## Train

python3 main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 50 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes 3

## Test for video classification output 
python3 inference.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes 3 --resume_path <model path>
    
## Get Saliency
python3 saliency.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes 3 --resume_path <model path>
    
## Plot Saliency 
python3 plotSaliency.py




