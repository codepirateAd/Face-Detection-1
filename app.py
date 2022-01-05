# Importing required libraries, obviously
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

html_string = '''<h3>Code to detect Faces from photos</h3><br>
                <p style="background-color:black;color:white;height:450px;width:600px;border: 1px solid black">
                import cv2
                #Load the cascade<br>
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')<br>
#Read the input image<br>
img = cv2.imread('test.png')<br>
#Convert into grayscale<br>
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
#Detect faces<br>
faces = face_cascade.detectMultiScale(gray, 1.1, 4)<br>
#Draw rectangle around the faces
for (x, y, w, h) in faces:<br>
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)<br>
#Display the output<br>
cv2.imshow('img', img)<br>
cv2.waitKey()</p><br>
<h4>Example output: &nbsp; <b>test.jpg<b></h4><br>
'''
st.markdown(html_string, unsafe_allow_html=True)
image = Image.open('outputeg.jpeg')
st.image(image)
# Loading pre-trained parameters for the cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    '''
    Function to detect faces/eyes and smiles in the image passed to this function
    '''
    
    image = np.array(image.convert('RGB'))
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)



    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        
        # The following are the parameters of cv2.rectangle()
        # cv2.rectangle(image_to_draw_on, start_point, end_point, color, line_width)
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]
        
        # Detecting eyes in the face(s) detected
        eyes = eye_cascade.detectMultiScale(roi)
        
        # Detecting smiles in the face(s) detected
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        
        # Drawing rectangle around eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        # Drawing rectangle around smile
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    # Returning the image with bounding boxes drawn on it (in case of detected objects), and faces array
    return image, faces


def about():
	st.write(
		'''
        Developer - CodePirate Ad
		''')


def main():
    st.title("Test the code ")
        
        # You can specify more file types below if you want
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file is not None:

    	image = Image.open(image_file)

    	if st.button("Process"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
    		result_img, result_faces = detect(image=image)
    		st.image(result_img, use_column_width = True)
    		st.success("Found {} faces\n".format(len(result_faces)))




if __name__ == "__main__":
    main()
