import cv2
from random import randrange

#using the cascade algorithm to make a classifier using the training data
preexisting_facedata = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

'''read image'''
#image = cv2.imread('mf.jpg')
'''haarcascade algrthm only takes grayscale faces'''
#grscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#capture webcam
vid = cv2.VideoCapture(0)

w = int(vid.get(3))
h = int(vid.get(4))
out = cv2.VideoWriter('voutput.mov', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))

while True:
    fr_read, frame = vid.read()
    grscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    detect face with classifier, multiscale means small or big
    will give us coordinates around face
    lol coordinates for bounding rectangle
    '''
    face_crdnts = preexisting_facedata.detectMultiScale(grscale_image)
    #print(face_crdnts) 

    #draw rectangle
    for (x, y, w, h) in face_crdnts:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1) #image, crdnts, rectangle color, thickness
    
    out.write(frame)

    cv2.imshow('image', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

vid.release()

print("code completed")
