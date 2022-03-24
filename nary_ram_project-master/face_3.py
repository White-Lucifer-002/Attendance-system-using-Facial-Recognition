import face_recognition
import imutils
import pickle
import time
import cv2
import os
import sys
from datetime import datetime
import csv
import pandas as ps

cols = ["S.no","Name","Attendance"]

rows = [["Sriram","A"],["Naresh","A"],["Aadhi","A"],["Gokul","A"],["Barath","A"],["Srijith","A"]]
rows.sort()
x = 1
for i in rows :
    i.insert(0,x)
    x += 1

cascPathface = os.path.dirname( cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)

#your specified file path
data = pickle.loads(open('face_enc', "rb").read())
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
    
                name = data["names"][i]
 
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)

        names.append(name)
        
        for x in rows:
            if x[1] == name :
                x[2] = "P"

 
        for ((x, y, w, h), name) in zip(faces, names):
 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

#your specified file path
with open('attendance.csv', 'w') as f:

    write = csv.writer(f)

    write.writerow(["Date :",str(datetime.today().strftime('%d-%m-%Y'))])
    write.writerow(cols)
    write.writerows(rows)
    f.close()

#your specified file path
data = ps.read_csv('attendance.csv')
print(data)
sys.exit()