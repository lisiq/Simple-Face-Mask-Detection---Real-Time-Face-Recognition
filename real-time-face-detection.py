import cv2
import os

# Locate the classifier
path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

# We load the classifier
faceCascade = cv2.CascadeClassifier(path)

cap = cv2.VideoCapture(0)

# Infinit while loop to get all the frames until we stop it
while True:
    # Get one frame per loop
    ret, frame = cap.read()
    # Classifier needs greyscale frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # We call the classifier
    faces = faceCascade.detectMultiScale(gray, 
        scaleFactor=1.2, # how much image size is reduced at each image scale
        minNeighbors=5,  # how many neighbors each candidate rectange should have
        minSize=(30, 30)) # minimum size of the detected face
        
    # We put a rectangle in the image when we detect a face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        # Display the resulting frame
    
    cv2.imshow('video', frame)
    k = cv2.waitKey(100) & 0xFF
    if k == 25: 
        break

cap.release()
cv2.destroyAllWindows()