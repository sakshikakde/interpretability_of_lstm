import numpy as np
import cv2 
import os
if __name__ == "__main__":
    image_folder_path = "./data/kth/image_data/"
    classes = os.listdir(image_folder_path)
    for c in classes:
        dataset = os.listdir(os.path.join(image_folder_path, c))
        for d in dataset:
            dataset_path = os.path.join(os.path.join(image_folder_path, c, d))
            print("......................Calculating flow for ", dataset_path)
            save_folder = os.path.join(os.path.join("./saliency/flow/", d))
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            images = [f for f in os.listdir(dataset_path) if f.endswith('jpg')]
            images = sorted(images, key=lambda x: int(x.split("_")[1].split(".")[0]))
            clip = []
            for image in images:
                image_file = os.path.join(dataset_path, image)
                frame = cv2.imread(image_file)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clip.append(frame)

            prev = clip.pop(0)
            bin_flow = np.zeros_like(prev)
            for i, next in enumerate(clip):
                flow = cv2.calcOpticalFlowFarneback(prev,
                                                    next, 
                                                    None, 
                                                    0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                bin_flow = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                file_name = "flow_"+ str(i) +".png"
                cv2.imwrite(os.path.join(save_folder, file_name), bin_flow)
                prev = next