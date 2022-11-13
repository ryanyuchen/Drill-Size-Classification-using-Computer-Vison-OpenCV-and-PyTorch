import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
    
def get_video_files(directory):
    videos = []
    for file in os.listdir(directory):
        videos.append(file)
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
        if num_images < 2:
            success, image = vidcap.read()
            num_images += 1
            label += 1
            file_name = name + "_" + str(label) + ".png"
            path = os.path.join(train_folder, file_name)
            cv2.imwrite(path, image)
            
            if cv2.imread(path) is None:
                os.remove(path)
            else:
                if not silent:
                    print(f'Image successfully written at {path}')
            
            count += delay*fps
            vidcap.set(1, count)
        elif num_images == 2:
            success, image = vidcap.read()
            num_images += 1
            label += 1
            file_name = name + "_" + str(label) + ".png"
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
        

def create_dataset(directory, h, w):
    img_data = []
    class_name = []
    
    for file in os.listdir(directory):
        img_path = os.path.join(directory, file)
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        #image = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (h, w), interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        img_data.append(image)
        class_name.append(file[0:5])
        
    return img_data, class_name
    
def plot_image(image):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    f.set_figwidth(15)
    ax1.imshow(image)
    
    # RGB channels
    # CHANNELID : 0 for Red, 1 for Green, 2 for Blue. 
    ax2.imshow(image[:, : , 0]) #Red
    ax3.imshow(image[:, : , 1]) #Green
    ax4.imshow(image[:, : , 2]) #Blue
    f.suptitle('Different Channels of Image')
