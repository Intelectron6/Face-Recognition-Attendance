import sys
import csv

if not sys.warnoptions:
    import warnings, os
    warnings.simplefilter("ignore")

from keras.models import load_model

classifier = load_model('face_classifier.h5')

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
from datetime import datetime
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
import pandas as pd
import time

def disp_percentage(df,i):
    cp = 0
    ca = 0
    track = list(df.iloc[i,1:])
    for x in track:
        if x == 'A':
            ca = ca+1
        elif x == 'P':
            cp = cp+1
    perc = (cp/(cp+ca))*100
    print("Your attendance is currently at", perc, "%")
    if perc < 75:
        print("You have shortage of attendance, please attend the class regularly")
    
    
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

def recognizer():
    df = pd.read_csv("database.csv")
    now = datetime.now()
    time1 = now.strftime("%d %B %H:00")
    if df.columns[-1] != time1:
        new_column = 'A'
        df['NewColumn'] = new_column
        df = df.rename(columns = {'NewColumn': time1})
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        #cv2.imshow('Live capture', frame)
        if face_extractor(frame) is not None:
            face = cv2.resize(face_extractor(frame), (200, 200))
            gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            cv2.imwrite('sample.jpg',gray)
            time.sleep(0.5)
            test_image = cv2.imread('sample.jpg')
            test_image = np.expand_dims(test_image, axis = 0)
            pred = classifier.predict(test_image)[0]
            
            for i in range(len(pred)):
                if pred[i] == max(pred) and pred[i]>0.95:
                    print(df.iloc[i,0]," present, updated")
                    df.iloc[i,-1] = 'P'
                    disp_percentage(df,i)
            time.sleep(0.5)

        if cv2.waitKey(1)==13:
            break
        else:
            print("Face not Found")
            pass
    cap.release()
    cv2.destroyAllWindows()
    df.to_csv('database.csv',index=False)

#recognizer()
