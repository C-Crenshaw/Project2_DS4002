# Source Code Folder for Project 2

Updated verisons of the source code are included in this file. Python file and Jupyter Notebook file types of "P2_Source_Code" are included. 


Within the files, the structure of the code is largely oriented around three sections: Installs and Imports, Detection Model with Single Items, and Detection Model with Multiple Items. Users should follow each section carefully to reproduce the desired outputs. This ReadMe includes a supplemental documentation to the main repository and gives a detailed account of each section's code. 


### Installs and Imports
Our model an engineered adaptation of the MediaPipe Object Detector. This model allows the user to detect the presence and location of multiple classes of objects in Python. In order to run this model, the following imports/installs must be loaded into the Python software: MediaPipe library, displaying imports, object detection imports, and MediaPipe "EfficientDet-Lite2 (int8)" model. 

Specifically with respect to the "efficientdet_lite2.tflite" file, the model will not run unless this object is in the files section in Google Colab/Python software/Jupyter Notebook. See the codeblock below. 

```
#Filepath for downloaded MediaPipe "EfficientDet-Lite2 (int8)" model
model_path = "/content/efficientdet_lite2.tflite"
```


###Detection Model: Single Images with Single Items


###Detection Model: Single Image with Multiple Items


The original file can be found on [Google Colaboratory](https://colab.research.google.com/drive/1YPt-yQZR_OffM-WwLsKOsho0I5PVS_6A?usp=sharing](https://colab.research.google.com/drive/14-p9glhQ1IrUiko6xtq0YeYRt7nQd3yb?usp=sharing)https://colab.research.google.com/drive/14-p9glhQ1IrUiko6xtq0YeYRt7nQd3yb?usp=sharing).  
