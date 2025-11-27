## Vehicle License Plate Detection ##
 The object detection model is based on the YoloV7 algorithm which focuses on recognizing the number plate of a vehicle-based. The model is based on YoloV7. The YoloV7 is not an official version and it is an improved version of the YoloV5 with much better performance in terms of accuracy. In this project, the steps involved are:-
1) Preparation of the dataset for training 
2) Cloning the YoloV7 project
3) Training the model based on the dataset
4) Testing the model

### Dataset preparation ###
The dataset for training the model is taken from [Kaggle](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection) and the open-source dataset for vehicle number plate recognition is difficult to obtain. The vehicle dataset has 225 Images with labels collected from different sources. the dataset is fairly good and the labels are based on XML format. So the dataset has done the preprocessing before being fed to the YoloV7 model. The bounding boxes of the labeled images are based on 4 points (Xmin, Xmax, Ymin, Ymax). But the YoloV7 requires bounding boxes in the format. **<class, center_x, center_y, width, height>**


<img src="/Tutorial Images/Yolov7 bounding box.png" alt="Alt text" title="YOLOV7 bounding box">

[Source: Paperspace](https://blog.paperspace.com/train-yolov7-custom-data/)



The whole datset is split into train, validation and test based on the proportion 80:10:10. The YoloV7 requires a particular structure of the train, test and calidation dataset. The folder format of the images and lables hould be like this:
```
train
│──Images
│──Labels   
test
│──Images
│──Labels 
val
│──Images
│──Labels 
```
Once the dataset is split into train, test, val with the lables, it'S time to train the model based on the dataset.

### Cloning the YoloV7 repository ###

As mentioned earlier, YoloV7 is not an official version and can be considered as an improved version of YoloV5.It has a better accuracy than YoloV7 and performs better.
``` 
git clone https://github.com/WongKinYiu/yolov7 
```
The corresponding libraries required for the YoloV7 can be installed with the help of the requirements. 

```
pip install -r ./requirements.txt
```

The python scripts for the model is created in google collab as YoloV7 requires the support of GPU for training. The dataset is uploaded to the google drive and then mounted the drive in the google collab. 

### Model Training ##

It is easy to train the model based on the pre trained model from the YoloV7 repository. The pre trained weight can be downloaded from the YoloV7 repo to start with. 
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```

The model can be trained with **train.py** with command line arguments.
```
python train.py --weights yolov7-e6e.pt --data data/licenseplate.yaml --cfg cfg/training/yolov7-num.yaml --batch-size 8 --name Model --epochs 200
```
The command line arguments are as follow:

 1) **weights** - The .pt file download from the YoloV7 reppository
 2) **data** - It is a file that has to be created and it contains the path of the tarin, validation and test images nad lables. It also has the number of classes. Here it is only 1 class i.e. license_plate
    ```
    train: ./train
    val: ./val
    test: ./test

    nc: 1
    names: [
        'license_plate'
    ]
    ```
  3) **cfg** - The ocnfig file also has to be changed based on the number of the class. It has the architecture of the model. Only the number of class has to be changed in the file.
  
  After training the model, it has to be tested to check whether the object of our focus are detected or not. Yolo generally has a good detection rate as compared to other algorithms and it can also used for real time object detection.

  ```
python detect.py --weights runs/train/Model/weights/best.pt --conf 0.4 --img-size 640 --source test/images --no-trace
```

The command line arguments are as follow:

 1) **weights** - The .pt file based on our trianed model. It wil be available in the folder - Model
 2) **source** - the location of the images which has to be tested.
 3) **conf** - The confidence score

 ## Result ##

 The object recognition model based on the YoloV7 has successfully detected the license plate of vehicles. Both Images and video was tested on the model. The Model is not perfect as the dataset comparatively less number of images and due to this, it will be probably underfitting.

[![Model Test on video](https://img.youtube.com/vi/C5QqRF5XooM/0.jpg)](https://youtu.be/C5QqRF5XooM "Model Test on Video file")

<img src="/Tutorial Images/N17.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N38.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N44.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N48.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N72.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N82.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N120.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N121.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N144.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N154.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N175.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N198.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N229.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N230.jpeg" alt="Alt text" title="YOLOV7 Training Results">
<img src="/Tutorial Images/N247.jpeg" alt="Alt text" title="YOLOV7 Training Results">

## Conclusions ##

The model is able to detect succesfully almost all the license plates. But there was some issues noticed from the model:-

1) The license plate with red background was not detected --> Not enough training data with red background.
2) Model may be overfitting after 200 epochs as it detects the naming of the car as number plates, for eg: with a confidence of 0.4 in one of the results
3) Diverese images has to be trained to get a better result, but it serves as a starting point to detect the number plates.
