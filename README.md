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

# Dataset 
## File structure
    .
    ├── data
    │   ├── raw_dataset
    │   │   ├── kth_data (copy the walking and running data here)
    │   │   │   ├── running 
    │   │   │   ├── walking
    │   ├── kth
    │   │   ├── annotations 
    │   │   ├── image_data
    │   │   │   ├── running
    │   │   │   ├── walking
    │   ├── kth_trimmed
    │   │   ├── running 
    │   │   ├── walking
    ├── model
    ├── utils
    └── other .py files
## Data generation
### Download
- The dataset can be downloaded from ![here](https://www.csc.kth.se/cvap/actions/). We are using only the walking and running classes. Copy it to the raw_data folder or follow the following steps.    
``` mkdir data ```   
``` cd data```      
``` mkdir kth ```   
``` mkdir raw_dataset```
``` cd raw_dataset ```    
``` wget http://www.nada.kth.se/cvap/actions/walking.zip ```    
``` wget http://www.nada.kth.se/cvap/actions/running.zip```

### Trimming the videos
- Run the following command from the root directory     
``` python3 utils/trim_videos.py ```
### Data preperation
- Run the followig command          
``` ./utils/generate_data.sh ```


## Download pre processed data
```wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sMn_BGhqmGdgZ0JKlodBwVB0Z5ppNqi_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sMn_BGhqmGdgZ0JKlodBwVB0Z5ppNqi_" -O data.zip && rm -rf /tmp/cookies.txt```       
```unzip data.zip```

# Models
The pretrained models can be downloaded from ![here](https://drive.google.com/file/d/11KT6b9pKAwP7zUBMyrJDlZRsG8qsTD5f/view?usp=sharingS -->).


# How to run code 
## Train

```python3 main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 50 --num_workers 0  --annotation_path ./data/kth/annotation/kth1.json --video_path ./data/kth/image_data/  --dataset kth --sample_size 150 --lr_rate 1e-4 --n_classes 2```

## Test for video classification output 
 ```python3 inference.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>```

## Get frame Saliency
If you want to use TSR, set a flag to true in frame_saliency.py      
```python3 frame_saliency.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path>```
    
## Get feature temporal Saliency
```python3 feature_saliency.py  --annotation_path ./data/kth/annotation/kth1.json  --dataset kth --model cnnlstm --n_classes 2 --resume_path <model_path> ```

# Results
## Temporal saliency





