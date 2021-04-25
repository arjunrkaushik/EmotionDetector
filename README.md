# EmotionDetector

This is a Deep Learning project which uses Python and OpenCV to detect emotions on human faces - angry/happy/sad/neutral/surprised.

## How To Run The Project 

Once you clone the repository, follow the below steps to get the Emotion Detector running.

**Step 1 -**

* Download the images(dataset) from https://drive.google.com/drive/folders/10rs9QzMICqkatzCBS2xFmecN5LmmZms8?usp=sharing
* Set the path for train_data_dir and validation_data_dir in the **trainer.py** file
* Run the trainer.py file in command prompt as **python trainer.py**(For Windows)
* Allow it to train - estimated time is a little over 3 hours
* A pre-trained file is available in case you are unable to train - **Emotion_little_vgg.h5**

**Step 2 -**
* Set the right path for face_classifier and classifier
* Run the image_detector.py file in command prompt as **python image_detector.py**

Voila, your Emotion Detector is up and running. 

## Instructions To Use(For Windows)

* Press **Esc** key to exit the program
* Press **s** to save a frame
