import numpy as np
import cv2
from glob import glob
import os

from numpy.ma.core import reshape


folder_path = "./data/raw_dataset/kth_data/"
data_save_path = "./data/kth_trimmed_data/"
if not os.path.exists(data_save_path):
    os.mkdir(data_save_path)

if not os.path.exists(data_save_path):
    os.mkdir(data_save_path)
folders = os.listdir(folder_path)
num_frames = 70
frame_size = 120

def getClassName(folder_path):
    tokens = folder_path.spilt('/')
    return tokens[-1]


for folder in folders:
    if folder == "running":
        file_path = os.path.join(folder_path, folder)
        save_folder_name = os.path.join(data_save_path, folder)
        files = os.listdir(file_path) 
        if not os.path.exists(save_folder_name):
            os.mkdir(save_folder_name)
        for file in files:
            print("running")
            cap = cv2.VideoCapture(os.path.join(file_path,file))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps =  cap.get(cv2.CAP_PROP_FPS)
            frame_start = 0
            video_write = cv2.VideoWriter(os.path.join(save_folder_name, str(frame_start) + "_"+ file), 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            fps, (frame_size, frame_size), 0)
            count = 0
            while(count < frame_start + num_frames):
                ret, frame = cap.read()
                if (frame_start <= count <= frame_start + num_frames):
                    if ret:
                        frame = cv2.resize(frame, (frame_size, frame_size))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        video_write.write(frame)
                count += 1
            video_write.release()
            cap.release()
    if folder == "walking":
        file_path = os.path.join(folder_path, folder)
        save_folder_name = os.path.join(data_save_path, folder)
        files = os.listdir(file_path) 
        if not os.path.exists(save_folder_name):
            os.mkdir(save_folder_name)
        for file in files:
            print("walking")
            cap = cv2.VideoCapture(os.path.join(file_path,file))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps =  cap.get(cv2.CAP_PROP_FPS)
            frame_start = 0
            video_write = cv2.VideoWriter(os.path.join(save_folder_name, str(frame_start) + "_"+ file), 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            fps, (frame_size, frame_size), 0)
            count = 0
            while(count < frame_start + num_frames):
                ret, frame = cap.read()
                if (frame_start <= count <= frame_start + num_frames):
                    if ret:
                        frame = cv2.resize(frame, (frame_size, frame_size))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        video_write.write(frame)
                count += 1
            video_write.release()
            cap.release()


            

