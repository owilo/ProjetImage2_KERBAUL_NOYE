import cv2
import os
import subprocess


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("video.mp4")
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

tracker = cv2.legacy.TrackerKCF_create()

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))
frame_number = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        if w > 0 and h > 0:  
            tracker.init(frame, (x, y, w, h))
    
    filename = f"frame_{frame_number}.png"
    cv2.imwrite(filename, frame)

    output_file = "frame_obscurcie.png"
    command = "../obscuration/pixelisation"+" "+filename+" "+output_file+" "+str(x)+" "+str(y)+" "+str(x+w)+" "+str(y+h)+" 20"          
    subprocess.run(command, shell=True, capture_output=False, text=False)

    frame_obscurcie = cv2.imread(output_file)
    os.remove(filename)
    
    out.write(frame_obscurcie)
    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()