# import all the libraries

from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import cv2
import os
import imutils
import numpy as np
from keras.models import load_model
import pickle
import dlib
import sys
import skimage 
from PIL import Image
import glob
from gtts import gTTS

def predict_eye(image,count,model,eyeLB):
    # Load the Haar cascade files for face and eye
    face_cascade = cv2.CascadeClassifier('eye_extraction/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('eye_extraction/haarcascade_eye.xml')

    # Check if the face cascade file has been loaded correctly
    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')

    # Check if the eye cascade file has been loaded correctly
    if eye_cascade.empty():
        raise IOError('Unable to load the eye cascade classifier xml file')

    # Initialize the object
    frame = image
    new_image = []
    # Define the scaling factor
    ds_factor = 0.5

    # Resize the frame
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Run the face detector on the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face that's detected, run the eye detector
    for (x,y,w,h) in faces:
        # Extract the grayscale face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Extract the color face ROI
        roi_color = frame[y:y+h, x:x+w]

        # Run the eye detector on the grayscale ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            new_image = roi_color[y_eye : y_eye+h_eye , x_eye : x_eye+w_eye]
            #cv2.imwrite("output/Eye_Detector.jpg", new_image)

    if len(new_image)==0:
        return "None",0.0
    new_image = cv2.resize(new_image, (64, 64))
    new_image = new_image.astype("float") / 255.0
    new_image = img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)

    # classify the input image using Keras' multi-output functionality
    #print("[INFO] classifying image...")
    prob = model.predict(new_image)
    #print("The predicted probability (eye) is ",prob)
    eyeLabel = eyeLB.classes_
    #print("The eye labels are : ",eyeLabel)
    #mouthText = "head: {} ({:.2f}%)".format(mouthLabel, mouthProba[0][mouthIdx] * 100)
    #cv2.putText(output, mouthText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    if prob<0.5:
        #print("Eye Prediction : ",eyeLabel[0],prob)
        return (eyeLabel[0],prob)
    else:
        #print("Eye Prediction : ",eyeLabel[1],prob)
        return (eyeLabel[1],prob)


def predict_mouth(image,count,model,mouthLB):
     # Load the Haar cascade files for face and eye
    face_cascade = cv2.CascadeClassifier('eye_extraction/haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('mouth_extraction/haarcascade_mcs_mouth.xml')

    # Check if the face cascade file has been loaded correctly
    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')

    # Check if the eye cascade file has been loaded correctly
    if mouth_cascade.empty():
        raise IOError('Unable to load the eye cascade classifier xml file')

    # Initialize the object
    frame = image
    new_image = []
    # Define the scaling factor
    ds_factor = 0.5

    # Resize the frame
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Run the face detector on the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face that's detected, run the eye detector
    for (x,y,w,h) in faces:
        # Extract the grayscale face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Extract the color face ROI
        roi_color = frame[y:y+h, x:x+w]

        # Run the eye detector on the grayscale ROI
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        count = 0
        for (x_eye,y_eye,w_eye,h_eye) in mouths:
            new_image = roi_color[y_eye : y_eye+w_eye , x_eye : x_eye+h_eye]
            cv2.imwrite("output_new/mouth_Detector_"+str(count)+".jpg", new_image)
            count = count+1
            break

        
    if len(new_image)==0:
        return "None",0.0
    new_image = cv2.resize(new_image, (64, 64))
    new_image = new_image.astype("float") / 255.0
    new_image = img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)

    
    # classify the input image using Keras' multi-output functionality
    #print("[INFO] classifying image...")
    prob = model.predict(new_image)
    #print("The predicted probability (mouth) is ",prob)
    mouthLabel = mouthLB.classes_
    #print("The mouth labels are : ",mouthLabel)
    #mouthText = "head: {} ({:.2f}%)".format(mouthLabel, mouthProba[0][mouthIdx] * 100)
    #cv2.putText(output, mouthText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    if prob<0.5:
        #print("Mouth Prediction : ",mouthLabel[0],prob)
        return (mouthLabel[0],prob)
    else:
        #print("Mouth Prediction : ",mouthLabel[1],prob)
        return (mouthLabel[1],prob)

    
def predict_head(new_image,model,headLB):
    new_image = cv2.resize(new_image, (64, 64))
    new_image = new_image.astype("float") / 255.0
    new_image = img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)

    
    # classify the input image using Keras' multi-output functionality
    #print("[INFO] classifying image...")
    prob = model.predict(new_image)
    #print("The predicted probability (head) is ",prob)
    headLabel = headLB.classes_
    #print("The head labels are : ",headLabel)
    #mouthText = "head: {} ({:.2f}%)".format(mouthLabel, mouthProba[0][mouthIdx] * 100)
    #cv2.putText(output, mouthText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    if prob<0.5:
        #print("Head Prediction : ",headLabel[0],prob)
        return (headLabel[0],prob)
    else:
        #print("Head Prediction : ",headLabel[1],prob)
        return (headLabel[1],prob)
    

def predict_face(new_image,model,faceLB):
    new_image = cv2.resize(new_image, (64, 64))
    new_image = new_image.astype("float") / 255.0
    new_image = img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis=0)

    
    # classify the input image using Keras' multi-output functionality
    #print("[INFO] classifying image...")
    prob = model.predict(new_image)
    #print("The predicted probability (head) is ",prob)
    faceLabel = faceLB.classes_
    #print("The head labels are : ",headLabel)
    #mouthText = "head: {} ({:.2f}%)".format(mouthLabel, mouthProba[0][mouthIdx] * 100)
    #cv2.putText(output, mouthText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    if prob<0.5:
        #print("Face Prediction : ",faceLabel[0],prob)
        return (faceLabel[0],prob)
    else:
        #print("Face Prediction : ",faceLabel[1],prob)
        return (faceLabel[1],prob)


##### WEBCAM

eye_model = load_model("output_new/drowsy_eye.model")
eyeLB = pickle.loads(open("output_new/eye_lb.pickle", "rb").read())

mouth_model = load_model("output_new/drowsy_mouth.model")
mouthLB = pickle.loads(open("output_new/mouth_lb.pickle", "rb").read())

head_model = load_model("output_new/drowsy_head.model")
headLB = pickle.loads(open("output_new/head_lb.pickle", "rb").read())

face_model = load_model("output_new/drowsy_face.model")
faceLB = pickle.loads(open("output_new/face_lb.pickle", "rb").read())

tts_1 = gTTS(text='ALERT!!! HEAD BENT', lang='en')
tts_1.save("head.mp3")

tts_1 = gTTS(text="WARNING!!! FACE COVERED", lang='en')
tts_1.save("face.mp3")

tts_1 = gTTS(text="WARNING!!! EYES CLOSED", lang='en')
tts_1.save("eyes.mp3")

tts_1 = gTTS(text="WARNING!!! MOUTH OPEN", lang='en')
tts_1.save("mouth.mp3")

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 0)
count = 0 

out = cv2.VideoWriter('/home/akansha/Desktop/output.avi', -1, 20.0, (640,480))


count_head=0
count_eye=0
count_face=0
count_mouth=0

while True:
    # Capture frame-by-frame
    count = count+1
    ret, image = video_capture.read()
    
    #(rows, cols) = image.shape[:2] 
    
    #image = image[0:(rows*3)//4 , 0:(cols*3)//4]
    
    #(rows, cols) = image.shape[:2] 
    #M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1) 
    #image = cv2.warpAffine(image, M, (cols, rows)) 

    head, head_prob = predict_head(image,head_model,headLB)
    eye, eye_prob = predict_eye(image,0,eye_model,eyeLB)
    mouth, mouth_prob = predict_mouth(image,0,mouth_model,mouthLB)
    face, face_prob = predict_face(image,face_model,faceLB)
    #image = cv2.resize(image,(800,800))
    
    if head_prob<0.5:
        count_head = count_head+1
    if head_prob>0.5:
        count_head = 0
    if face_prob<0.5:
        count_face = count_face+1
    if face_prob>0.5:
        count_face = 0
    if eye_prob<0.5 and eye_prob>0:
        count_eye = count_eye+1
    if eye_prob>0.5:
        count_eye=0
    if mouth_prob>0.5:
        count_mouth = count_mouth+1
    if mouth_prob<0.5:
        count_mouth=0
    
    if count_head > 10:
        cv2.putText(image, "ALERT!!! HEAD BENT", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 2)
        os.system("start head.mp3")
    elif count_face > 10:
        cv2.putText(image, "WARNING!!! FACE COVERED", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 2)
        os.system("start face.mp3")
    elif count_eye > 10:
        cv2.putText(image, "WARNING!!! EYES CLOSED", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 2)
        os.system("start eyes.mp3")
    elif count_mouth > 10:
        cv2.putText(image, "WARNING!!! MOUTH OPEN", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 2)
        os.system("start mouth.mp3")
        
    if count_eye > 30:
        count_eye = 0
    if count_mouth > 30:
        count_mouth = 0
    if count_head > 30:
        count_head = 0
    if count_face > 30:
        count_face = 0

    cv2.putText(image, "HEAD : "+head+"_"+str(head_prob), (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(image, "EYE : "+eye+"_"+str(eye_prob), (10, 55), cv2.FONT_HERSHEY_SIMPLEX,0.7, ( 255,0, 0), 2)
    cv2.putText(image, "MOUTH : "+mouth+"_"+str(mouth_prob), (10, 85), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0, 255), 2)
    cv2.putText(image, "FACE : "+face+"_"+str(face_prob), (10, 115), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    
    cv2.imwrite("output_img/output_"+str(count)+"_img.jpg",image)

    # Display the resulting frame
    cv2.imshow('Video', image)
    out.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()


