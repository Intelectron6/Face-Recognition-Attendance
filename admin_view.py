import sys

if not sys.warnoptions:
    import warnings, os
    warnings.simplefilter("ignore")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
import pandas as pd
import os
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


df = pd.read_csv("database.csv")
n = len(df)
#print(n)

def face_extractor(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray,1.3,5)

	if faces is():
		return None
	
	for(x,y,w,h) in faces:
		cropped_face = img[y:y+h, x:x+w]

	return cropped_face

def train(n):
	classifier = Sequential()

	classifier.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))

	classifier.add(Flatten())

	classifier.add(Dense(units = 128, activation = "relu"))
	classifier.add(Dense(units = 128, activation = "relu"))
	classifier.add(Dense(units = n, activation = "softmax"))

	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

	from keras.models import load_model
	classifier.save('face_classifier.h5')

	import numpy as np
	from keras.preprocessing.image import ImageDataGenerator
	import cv2

	face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
		'Faces/train',
		target_size=(200, 200),
		batch_size=4,
		class_mode='categorical')

	validation_generator = test_datagen.flow_from_directory(
		'Faces/test',
		target_size=(200, 200),
		batch_size=4,
		class_mode='categorical')

	classifier.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=5,
		validation_data=validation_generator,
		validation_steps=20)

def add_new_faces(n):
    n = n+1
    nam = input('Enter the name of the student: ')
    new_row = {'Name' : nam}
    df = df.append(new_row, ignore_index=True)
    df.to_csv('database.csv',index=False)
    cap = cv2.VideoCapture(0)
    count = 0
    os.mkdir("Faces/train/user"+str(n))
    os.mkdir("Faces/test/user"+str(n))
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            file_name_path = 'Faces/train/user'+str(n)+'/'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==80:
            count = 0
            break

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = 'Faces/test/user'+str(n)+'/'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==20:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Added")

#add_new_faces(n)
#train(n)





