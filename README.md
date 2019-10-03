# Face-Recognition-Attendance
An automatic attendance system that uses face recognition to update attendance of individuals in the database.

Dependencies-
OpenCV, Keras, Pandas, Numpy and some basic libraries.

Steps to use:

1) Open system.py

2) Enter admin configuration. This will allow you to add face samples into the database. You will need to be in front of your
laptop's webcam for it take images and store them.

3) Once samples of all students/workers is added, train the convolutional neural network model.
 
4) After the training, the system can be used to update attendance every day in user configuration. 
The data will be stored in "database.csv". You just have to go to system.py and open user configuration and wait for the camera
to detect your face and recognize it. It will also display your attendance status.

5) More new users can be added using admin configuration, which will required you to retrain the CNN model.
