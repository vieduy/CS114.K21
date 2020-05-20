# USAGE
# python gather_selfies.py --face-cascade cascades/haarcascade_frontalface_default.xml \
#	--output output/faces/adrian.txt

# import the necessary packages
from __future__ import print_function
from scripts.face_recognition.facedetector import FaceDetector
from imutils import encodings
import argparse
import imutils
import cv2
from imutils import paths

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face-cascade", required=True, help="path to face detection cascade")
ap.add_argument("-o", "--output", required=True, help="path to output file")
ap.add_argument("-w", "--write-mode", type=str, default="a+", help="write method for the output file")
ap.add_argument("-d", "--dataset", required=True, help="path to Faces dataset")
args = vars(ap.parse_args())

# initialize the face detector, boolean indicating if we are in capturing mode or not, and
# the bounding box color
fd = FaceDetector(args["face_cascade"])
color = (0, 255, 0)

# grab a reference to the webcam and open the output file for writing
f = open(args["output"], args["write_mode"])
total = 0
i = 0
imagePaths = list(paths.list_images(args["dataset"]))
# loop over the frames of the video
for (i, imagePath) in enumerate(imagePaths):
    frame = cv2.imread(imagePath)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100))

    # ensure that at least one face was detected
    if len(faceRects) > 0:
        # sort the bounding boxes, keeping only the largest one
        (x, y, w, h) = max(faceRects, key=lambda b: (b[2] * b[3]))

        face = gray[y:y + h, x:x + w].copy(order="C")
        cv2.imshow('faces', face)
        cv2.waitKey(0)
        f.write("{}\n".format(encodings.base64_encode_image(face)))
        total += 1

    # show the frame and record if the user presses a key
    cv2.imshow("Frame", frame)


# close the output file, cleanup the camera, and close any open windows
print("[INFO] wrote {} frames to file".format(total))
f.close()
cv2.destroyAllWindows()
