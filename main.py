# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:40:55 2022

@author: Yu Chen
"""

from utils import get_videos, extract_images_from_video

def main():
    files = get_videos("VideoData")
    for file in files:
        path = "VideoData" + '/' + file
        name = file.split(".")[0]
        extract_images_from_video(video=path, name=name)
    
    

        

if __name__ == '__main__':
    main()