from utils import get_video_files, extract_images_from_video, create_dataset, plot_image
from models import cnn
from loss import FocalLoss, reweight

def main():
    # extra image from video and split into train and test
    files = get_video_files("VideoData")
    for file in files:
        path = "VideoData" + '/' + file
        name = file.split(".")[0]
        extract_images_from_video(video=path, name=name)
    
    # create train and testing dataset
    path_train = "ImageData/train"
    X_train, y_train = create_dataset(path_train, 1280//2, 960//2)
    #plot_image(X_train[0])
    #print(X_train[0][1:10,:,:])
    #print(y_train[0])
    
    path_test = "ImageData/test"
    X_test, y_test = create_dataset(path_test, 1280//2, 960//2)

        

if __name__ == '__main__':
    main()