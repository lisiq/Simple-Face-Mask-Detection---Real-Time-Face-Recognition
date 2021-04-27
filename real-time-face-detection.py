import cv2
import os

# Locate the classifier
path_face = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

# We load the classifier
faceCascade = cv2.CascadeClassifier(path_face)

noseCascade = cv2.CascadeClassifier('Nose.xml')

mouthCascade = cv2.CascadeClassifier('Mouth.xml')



cap = cv2.VideoCapture(0)

# Infinit while loop to get all the frames until we stop it
while True:
    # Get one frame per loop
    ret, frame = cap.read()
    # Classifier needs greyscale frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # We call the classifier
    faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, # How much image size is reduced at each image scale
        minNeighbors=5,  # How many neighbors each candidate rectange should have
        minSize=(30, 30)) # Minimum size of the detected face
        
    # We put a rectangle in the image when we detect a face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2) # Display the resulting frame
        # Are needed for the other classifiers
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # We call the classifier
        noses = noseCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3, # How much image size is reduced at each image scale
            minNeighbors=5, # How many neighbors each candidate rectange should have
            minSize=(15,15) # Minimum size of the detected face
            )

        # We put rectangle in the image when we detect nose
        for (xn,yn,wn, hn) in noses:

            if wn > 0 :
                cv2.rectangle(roi_color, (xn, yn), (xn + wn, yn +hn), (255, 0, 0), 2) # Display the resulting frame
                font = cv2.FONT_HERSHEY_SIMPLEX # Select font for the text
                cv2.rectangle(frame, (0, 0), (170,  50), (0,0,0), -1) # Make a black box at the top right corner
                cv2.putText(frame, 'Mask is OFF!',(int(150/10),int(50/2)), font, 0.7, (255,255,255)) # Display white text above balack box

        # We call the classifier
        mouths = mouthCascade.detectMultiScale(
            roi_gray,
            scaleFactor=4, # How much image size is reduced at each image scale
            minNeighbors=5, # How many neighbors each candidate rectange should have
            minSize=(25,25) # Minimum size of the detected face
            )

        # We put rectangle in the image when we detect nose
        for (xm, ym, wm, hm) in mouths:
            if wm > 0:
                cv2.rectangle(roi_color, (xm, ym), (xm + wm, ym + hm), (0,0,255), 2) # Display the resulting frame
                font = cv2.FONT_HERSHEY_SIMPLEX # Select font for the text
                cv2.rectangle(frame, (0, 0), (170,  50), (0,0,0), -1) # Make a black box at the top right corner
                cv2.putText(frame, 'Mask is OFF!',(int(150/10),int(50/2)), font, 0.7, (255,255,255)) # Display white text above balack box


        cv2.imshow('Do you have your mask on?', frame)
    
    # Break the loop by typing letter q
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()