# Drill Size Classification

The ML model used to classify the drill size is developed using Python, OpenCV and PyTorch. 

## Required Dependencies

- Please install the Python, OpenCV and PyTorch in order to use the application. 

## Files Structure

```
├── README.md
├── main.py
├── utils.py
├── models
│   └── cnn.py
├── loss
│   └── focal_loss.py
├── ImageData
│   ├── train
│   └── test
└── VideoData
```

## Steps

- Put videos in the directory of `VideoData`.
- The functions of `get_video_files` and `extract_images_from_video` in utils.py can extract images from video and put first two images in `ImageData\train` and the third image in `ImageData\test`. The images are in `.png` format.
- The function of `create_dataset` preprocesses the images and create two datasets for training and testing and also outputs the related labels.
- Unique labels are stored in a dictionary and covert the string label into numerical label.
- Hyperparameters of `batch_size`, `learning_rate`, `reg` and `epoch` are defined.
- A CNN model is developed and stored in the dirctory of `model` and the configuration is as follows: Image Data -> Conv Layer 1 (3, 16, 3, 1, 1) -> Batch Norm -> ReLU -> Conv Layer 2 (16, 32, 3, 1, 1) -> Batch Norm -> ReLU -> Max Pooling (2, 2, 0) -> Conv Layer 3 (32, 64, 3, 1, 1) -> Batch Norm -> ReLU -> Max Pooling (2, 2, 0) -> Dropout 1 (p=0.05) -> Conv Layer 4 (64, 128, 3, 1, 1) -> Batch Norm -> ReLU -> Max Pooling (2, 2, 0) -> Dropout (p=0.1) -> FC Layer 1 (2048, 1024) -> ReLU -> FC Layer 2 (1024, 512) -> ReLU -> Dropout 3 (p=0.1) -> FC Layer 3 (512, 10) -> Output.
- Optimizer is set to Adam
- Criterion initially uses `Cross-Entropy` and another loss fuction of `Focal-Loss` is developed and placed in the directory of `loss`
- In each epoch, the model is trained and validate and the best accuracy is used to evalute the model

## Future Improvement

- Hyperparamter tuning on batch size, learning rate and regularization
- Explore Focal Loss, which will put more weight on the difficult examples 
- The dataset is small, so we can use Image Augmentation to create more datas
- Explore RNN and LSTM

