"""
Created on Sat Nov 12 15:43:32 2022

@author: vanch
"""

import cv2
import os
    
def get_video_files(directory):
    videos = []
    for filename in os.listdir(directory):
        videos.append(filename)
        #print(filename)
        
    return videos

def extract_images_from_video(video, train_folder='ImageData/train', test_folder='ImageData/test', delay=10, name="file", max_images=5, silent=False):    
    vidcap = cv2.VideoCapture(video)
    count = 0
    num_images = 0
    label = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    
    while(success):
        if num_images < 3:
            success, image = vidcap.read()
            num_images += 1
            label += 1
            file_name = name + "_" + str(label) + ".jpg"
            path = os.path.join(train_folder, file_name)
            cv2.imwrite(path, image)
            
            if cv2.imread(path) is None:
                os.remove(path)
            else:
                if not silent:
                    print(f'Image successfully written at {path}')
            
            count += delay*fps
            vidcap.set(1, count)
        elif num_images == 3:
            success, image = vidcap.read()
            num_images += 1
            label += 1
            file_name = name + "_" + str(label) + ".jpg"
            path = os.path.join(test_folder, file_name)
            cv2.imwrite(path, image)
            
            if cv2.imread(path) is None:
                os.remove(path)
            else:
                if not silent:
                    print(f'Image successfully written at {path}')
            
            count += delay*fps
            vidcap.set(1, count)
        else:
            success = False
        