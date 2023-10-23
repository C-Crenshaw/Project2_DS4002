# Determining Product Sales and Inventory Trends: Analyzing Grocery Retailer Image Data
Group 10 <br />
**Project 1 Leader:** Carson Crenshaw <br />
**Members:** Sujith Panchumarthy, Zoe Averill <br />
DS 4002 <br />
Last Updated: 10/22/2023

## Description

MI3 deliverable for the second project cycle of DS 4002. <br />

The objective of this project is to build a Python image recognition model which can sufficiently analyze image data produced by grocery store chains, hypothesizing that an accurate model will be able to identify the products within each image and generate a reasonable total price for a given basket of items. This project group will attempt to fulfill this objective by answering the following questions: To what extent does the image recognition model accurately determine the grocery products within a given image? Will the similarity of appearance between some grocery items obscure their identification? How can data generated from check-out cameras be used to learn more about customer preferences and even track revenue generation? The overall success of the constructed model will be calculated in order to apply the model to more complex images of multiple products, simulating a customer’s purchasing behavior at checkout. 

## Contents of the Repository

The contents of this repository include the source code, data, figures, license, and references for MI3. 

## [SRC](https://github.com/C-Crenshaw/Project2_DS4002/tree/fc050f93dcebb51567d5ef2890a21deff47ea505/SRC)

This section contains all the source code for the project. All code has been consolidated in one python/jupyter notebook file. The original source code can be found [online](https://colab.research.google.com/drive/14-p9glhQ1IrUiko6xtq0YeYRt7nQd3yb?usp=sharing). 

### Installing/Building Code

* The code can be downloaded from the SRC folder in either python or Jupyter Notebook format. The finalized and most updated version of the code file can be found online on Google Colaboratory. The link to this file is shared above.

### Usage

The usage section focuses on the most complex aspects of the final code, highlighting the most important aspects of the final code. A supplemental documentation of the code usage can be found in the SRC folder on the main repository page. 

* This section describes key steps for setting up the development environment and code projects specifically to use Object Detector. All necessary imports are loaded into the 'P2_Source_Code.py' file. These necessary packages are repeated below. These imports include those standard to most Python projects and the MediaPipe packages utilized for the construction and testing of an Object Detection model. 
```
#Standard imports
import pandas as pd
import numpy as np
from google.colab import files

#Install MediaPipe library
!pip install mediapipe

#Image displaying imports
import cv2
from google.colab.patches import cv2_imshow

#Object detection imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Filepath for downloaded MediaPipe "EfficientDet-Lite2 (int8)" model
model_path = "/content/efficientdet_lite2.tflite"
```

* The object detection function of the code operates on image data with a machine learning (ML) model, accepting static data as input and outputting a list of detection results. Each detection result represents an object that appears within the image. For images with a single object, the detection results are appended into a data table which clearly illustrates the accuracy of the model. The following codeblock demonstrates the process of feeding the model images by manually selecting files from a computer drive. Once the files have been uploaded, the code creates a function for outlining and displaying the detected objects in an image and stores the detection results of the function in a dataframe. The final accuracy dataframe is then displayed and exported.
```
#Images with Single Items
#Image uploads
#Select all images from computer files
uploaded = files.upload()

for filename in uploaded:
  content = uploaded[filename]
  with open(filename, 'wb') as f:
    f.write(content)

#Record all image filenames
filenames = []
for filename in uploaded:
  print(filename)
  filenames.append(filename)

#Function for outlining and displaying the detected objects in an image
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    #Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    #Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

#Create an ObjectDetector object
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#Dataframe to store initial, single item, detection results (NOT for basket images with multiple items)
accuracyDF = pd.DataFrame(columns=["File Name","Object","Detected?","Confidence","Accepted?"])

#Loop through uploaded files
for IMAGE_FILE in filenames:
  #Load the input image
  image = mp.Image.create_from_file(IMAGE_FILE)

  #Detect objects in the input image
  detection_result = detector.detect(image)

  #Get object name for accuracy calculations
  objectName = IMAGE_FILE.rsplit("_")[0]

  #Flag to check if object is detected
  flag = False
  #If objects are detected, loop through then
  if len(detection_result.detections) > 0:
    for detection in detection_result.detections:
      #If object is correctly detected, check confidence
      if objectName == detection.categories[0].category_name:
        #Mark that object was detected
        flag = True
        #If confidence is greater than or equal to 75%, object is considered detected
        if detection.categories[0].score >= 0.75:
          row = [IMAGE_FILE,objectName,"Yes",detection.categories[0].score,"Yes"]
          break
        #If confidence is less than than 75%, object is NOT considered detected
        else:
          row = [IMAGE_FILE,objectName,"Yes",detection.categories[0].score,"No"]
          break

  #If no objects were detected or none of the detected objects were the object, record that object wasn't detected
  if flag == False:
    row = [IMAGE_FILE,objectName,"No",np.nan,np.nan]

  #Append row to accuracy df
  accuracyDF.loc[len(accuracyDF)] = row

accuracyDF

#Exporting accuracy df
accuracyDF.to_csv('accuracy.csv', index=False)
```

* For images with multiple objects, the detection results are not appended into a data table. Instead, the code displays an annotated image result that illustrates whether the model successfully identified the grocery objects in the image. 
```
#Images with Multiple Items
#Image uploads
#Select all images from computer files
uploaded = files.upload()

for filename in uploaded:
  content = uploaded[filename]
  with open(filename, 'wb') as f:
    f.write(content)

#Record all image filenames
filenames = []
for filename in uploaded:
  print(filename)
  filenames.append(filename)

#Function for outlining and displaying the detected objects in an image
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    #Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    #Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

#Create an ObjectDetector object
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

#Load the input image
image = mp.Image.create_from_file("Project 2 DS4002.jpeg")

#Detect objects in the input image
detection_result = detector.detect(image)

#Display the detection result
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb_annotated_image)
```
