from imutils import paths
import face_recognition
import pickle
import cv2
import os
 
#your specified file path
imagePaths = list(paths.list_images('D:\\CIT\\Sem-4\\ML\\Lab\\Project\\Data\\Images'))
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}

#your specified file path
f = open("D:\\CIT\\Sem-4\\ML\\Lab\\Project\\Data\\face_enc", "wb")
f.write(pickle.dumps(data))
f.close()