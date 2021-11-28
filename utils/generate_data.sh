# rm -rf ./data/annotation
# mkdir ./data/annotation
# rm -rf ./data/image_data
# mkdir ./data/image_data

# python3 utils/video_jpg_ucf101_hmdb51.py
# python3 utils/n_frames_ucf101_hmdb51.py
# python3 utils/gen_anns_list.py
# python3 utils/ucf101_json.py

#for kth dataset
rm -rf ./data/kth/annotation
mkdir ./data/kth/annotation
rm -rf ./data/kth/image_data
mkdir ./data/kth/image_data

python3 ./utils/kth/video_jpg_kth_hmdb51.py
python3 ./utils/kth/n_frames_kth_hmdb51.py
python3 ./utils/kth/gen_anns_list.py
python3 ./utils/kth/kth_json.py
