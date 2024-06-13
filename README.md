# <center> Meters Reader (using YOLO and OpenCV) </center>  

## Table of Contents <a name = 'content'></a>  

1. [Problem statement](#problem)   
2. [Project navigation](#navigation)   
3. [Description of solution logic](#logic)    
4. [Example of a meter read](#performance_example)     
5. [How to run code on your device](#run_code)    
6. [Usage](#usage)    


## Problem statement <a name = 'problem'></a>

[Table of Contents](#content)


The goal of this project is to read meter readings from a photo using **YOLOv8** and **OpenCV** library. To solve this task, I used a [dataset from Kaggle](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset) that contains 1244 meter images and masks. The sample data from this set looks like this:   

<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/img_exml.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 100px;">Image example</p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/mask_exml.jpg" alt="Image 2" style="width: 300px;"/>
    <p style="margin-left: 100px;">Mask example</p>
  </div>

</div>   


## Project navigation <a name = 'navigation'></a>  

[Table of Contents](#content)   

* `data` folder contains original dataset from Kaggle, data prepared for training YOLO segmentation and detection models, data of each preparation step, annotation labels for detection model training and my meter photos that I used to test program;
* `models` folder contains trained segmentation and detection models, that are neccessary for making predictions;
* `pics` folder contains the supporting images for this notebook;
* `predictions` folder contains couple of examples of how program reads meter readings;    
* `utils.py` file contains all the functions that I use for data preparation, models training and making prediction;  
* `main.py` file perform data preparations, models training and making prediction using functions from the `utils.py`;
* `yolo_segment.yaml` is the file that is needed for the YOLO segmentation model to understand where to look for training data. It contains directory paths and classes;
* `yolo_detect.yaml` is the file that is needed for the YOLO detection model to understand where to look for training data. It contains directory paths and classes;
* `yolov8n-seg.pt` is a YOLO segmentation model with trained weights that I use as the backbone;
* `yolov8s.pt` is a YOLO detection model with trained weights that I use as the backbone;
* `requirements.txt` is a list of required libraries and their versions.


## Description of solution logic <a name = 'logic'></a>  

[Table of Contents](#content)

There are 3 main steps in the solution:
1) [Training the segmentation model to identify the area with numbers in the meter image](#segment);
2) [Training an object detection model to classify each number in that zone](#detect);    
3) [Converting the prediction of the object detection model to the meter reading and saving the results](#pred2num)   


### 1. Training the segmentation model <a name = 'segment'></a>        

[Back to description plan](#logic)

As mentioned, I used the YOLOv8 models by ultralytics. In order to achieve faster training and higher model accuracy, I used transfer learning approach.   

Before training the YOLO segmentation model, there are a few preparatory steps to follow.  


First I converted the data structure to the form that YOLO requires. You can see the schema below:   

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/dir_struc.jpg" alt="Image 1" style="width: 200px;"/>
    <p style="margin-left: 400px;"></p>
</div>
 

Next, I converted the masks to label format. This is a format that is also required for YOLO to work correctly.    

The last step in the preparation is to create a `yolo_segment.yaml` file. This contains path information for training, validation, and test data that helps YOLO find the data in the file system. The `.yaml` file looks like this:    

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/segment_yaml.jpg" alt="Image 1" style="width: 700px;"/>
    <p style="margin-left: 400px;"></p>
</div>


After all the preparations are done, I loaded the model `yolov8n-seg.pt` with trained weights. This is my backbone. Next, I trained the model on meter images for 10 epochs, and took the `best.pt` model to use.   

Resulting segmentation model gives such results: 


<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/img_exml.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 50px;">Image before segmentation</p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/segmented.jpg" alt="Image 2" style="width: 300px;"/>
    <p style="margin-left: 100px;">Segmented image</p>
  </div>

</div>    


### 2. Training the object detection model <a name = 'detect'></a>        

[Back to description plan](#logic)

The preparations for training the object detection model are more complicated. I need to get same numbers format on each meter in order to train a versatile model.    
  


First, I apply segmentation masks to the meter images and save the results to the `data/masked_by_YOLO` folder.      

Example of masked image:   

<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/img_exml.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 90px;">Original image</p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/masked.jpg" alt="Image 2" style="width: 300px;"/>
    <p style="margin-left: 80px;">Image masked by YOLO</p>
  </div>

</div>


Now my task is to carefully cut out the area with numbers.   
To do this, I first find the contour of the area with the numbers: 

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/numbers_contour.jpg" alt="Image 1" style="width: 400px;"/>
    <p style="margin-left: 375px;">Found numbers contour</p>
</div>   

I then find the minimum rectangle that bounds this contour:    
<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/rectangle.jpg" alt="Image 1" style="width: 400px;"/>
    <p style="margin-left: 360px;">Found bounding rectangle</p>
</div>  


Now I find the vertices of this rectangle in order *top left -> top right -> bottom left -> bottom right*:       

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/verticles.jpg" alt="Image 1" style="width: 400px;"/>
    <p style="margin-left: 400px;">Found verticles</p>
</div> 


Now I check what is bigger: the distance between vertices 0 and 1 or between vertices 0 and 2 (is numbers zone rotated horizontally or vertically). Based on the results of the check, I rotate the image by stretching the corners of the rectangle to the corners of the image and resize it.   

The logic of the operation looks like this:   

<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/hor_marks.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 60px;">Horizontally rotated numbers</p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/ver_marks.jpg" alt="Image 2" style="width: 225px;"/>
    <p style="margin-left: 40px;">Vertically rotated numbers</p>
  </div>

</div>

**Pay attention to where the corners of the number area "go"!**   


After rotating with this strategy and resizing, I got cut out areas with numbers in the same format:  

<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/cropped_hor.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 25px;">Cropped numbers from left image above</p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/cropped_ver.jpg" alt="Image 2" style="width: 300px;"/>
    <p style="margin-left: 25px;">Cropped numbers from right image above</p>
  </div>

</div>


Cropped images are saved in `data/cropped_by_YOLO` folder   


Unfortunately, the original dataset don't have the annotated data needed to train the number classifier, so I had to do everything manually. I used the free service [CVAT.ai](https://www.cvat.ai/) to annotate the data. On each image, I marked out the area where the digit is located and assigned each digit a class from 0 to 9. In total, just over 9,000 figures were noted.   

An example of the image annotation looks like this. Here, each color corresponds to a specific class:   

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/annot_exml.jpg" alt="Image 1" style="width: 400px;"/>
    <p style="margin-left: 370px;">Example of annotation</p>
</div> 

*I chose a criterion for data annotation - if I can definitely recognize a digit, I annotate it, even if only part of the digit is visible.*

Some images turned out upside down. This was due to the unusual angle of the meter photo. Here is an example of such an image:   

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/upside-down.jpg" alt="Image 1" style="width: 400px;"/>
    <p style="margin-left: 340px;">Example of upside-down image</p>
</div> 

I ignored the upside down images, as they are an absolute minority.   

I uploaded the annotated data in the format required for YOLO to work. It is important to remember that annotated data saved in `data/annotated_numbers_for_YOLO_detection/labels` is only suitable for data segmented with `yolov8n-seg.pt` for 10 epochs (`best.pt`).     


After that, I put the classifier training data into the same structure as for the segmentation model:  


<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/dir_struc.jpg" alt="Image 1" style="width: 200px;"/>
    <p style="margin-left: 400px;"></p>
</div>
  

The last step in the preparation is to create a `yolo_detect.yaml` file. This contains path information for training, validation, and test data that helps YOLO find the data in the file system. The `.yaml` file looks like this:    

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/detect_yaml.jpg" alt="Image 1" style="width: 550px;"/>
    <p style="margin-left: 400px;"></p>
</div>   


All preparations are made. Now I train `yolov8s.pt` for 20 epochs, and take the `best.pt` model for predictions.   


### Object detection model prediction to meter readings <a name = 'pred2num'></a>  

[Back to description plan](#logic)

The last thing left to do is to convert the model prediction into a number (meter readings).  

To do this, I sort the detected digits in order from left to right and remove all zeros from the beginning. 

It happens that the model recognizes two digits on the same vertical level. This happens most often on the last digit - where two digits can be clearly visible, as in the example below:    

<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/duplicate_num.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 100px;">Cropped image</p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/duplicate_det.jpg" alt="Image 2" style="width: 300px;"/>
    <p style="margin-left: 75px;">Detection model prediction</p>
  </div>

</div>   

I leave the digit that has the larger height of the bounding box. Since the height of the digit is greater, it is more visible and it is a more honest meter reading.     
In this particular example, the height of the bounding box of the 2 is greater than the height of the bounding box of the 1, so we leave the 2.   


The last issue to solve is the floating comma. Meter readings have integer and fractional parts, but there is no single variant of how they are visually separated. They can be separated by a comma, by the color of the digits, by the color of the background, by different fonts, so it's not clear how to create a universal recognition algorithm.   


However, I noticed that meters fall into two types: either they have only 5 digits and no fractional part, or they have more than 6 digits and the fractional part takes 3 digits.  

Two types of meters:

<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/3_float.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 40px;">More than 6 digits, 3 digits fract. part</p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/5_int.jpg" alt="Image 2" style="width: 300px;"/>
    <p style="margin-left: 85px;">5 digits, no fract. part</p>
  </div>

</div>   

I don't know if this rule works on all the meters in the world, but in my dataset of 1244 images, there is not a single violation of this rule. Moreover, the meters in my house also follow it. You can check the meters in your home)   

So I just check how many digits the program detected. If it is less than 6, I take the reading as a integer. If there are more than 6 digits, I divide the integer by 1000, shifting the comma by 3 digits.   

And that's it. The project is done!    


## Example of a meter read <a name = 'performance_example'></a>  

[Table of Contents](#content) 

Below you can see examples of how the program works on the photos of the meters from my apartment:   

<div style="display: flex; justify-content: space-around;">

  <div style="text-align: center; margin-right: 20px; margin-left: 150px;">
    <img src="pics/pred1.jpg" alt="Image 1" style="width: 300px;"/>
    <p style="margin-left: 40px;"></p>
  </div>
  
  <div style="text-align: center; margin-left: 20px;">
    <img src="pics/pred2.jpg" alt="Image 2" style="width: 300px;"/>
    <p style="margin-left: 85px;"></p>
  </div>

</div>     


## How to run code on your device <a name = 'run_code'></a>  

[Table of Contents](#content) 

To install all required dependencies, run the `requirements.txt` file from the root directory of the project:    
> pip install -r requirements.txt   

The `main.py` file contains variables that you can set to `True` to reproduce by yourself *data preparation for segmentation model training, segmentation model training, data preparation for object detection model training, object detection model training and prediction on the meter image.*     

<div style="text-align: center; margin-right: 20px; margin-left: 70px;">
    <img src="pics/hyper_vars.jpg" alt="Image 1" style="width: 550px;"/>
    <p style="margin-left: 400px;"></p>
</div>

If you want to train models on your own, remember:   
* Data for training segmentation model is stored in `data/YOLO_segmentation_data`;
* Data for training object detection model is stored in `data/YOLO_detection_data`;
* Replace the `path` variable in `.yaml` files with the **absolute** path to `YOLO_segmentation_data` and `YOLO_detection_data` in your file system;
* If you don't have GPU on your computer and you don't want training to last forever, use GoogleColab or something similar.      


## Usage <a name = 'usage'></a>  

[Table of Contents](#content)  

If you want to read the meter readings from your image, here's what you need to do:  
1) Set `MAKE_PREDICTION_ON_IMAGE` in `main.py` to `True`;
2) Replace the values of the `image_path` and `path_to_save_predictions` variables (lines 88 and 91 in `main.py`) with the path to your image and the path where you want to save the prediction, respectively;
3) Execute the `main.py` file:   
> python main.py   


I hope you like the solution. Enjoy coding and have a good day!   