# import the necessary packages
from scripts.descriptors.hog import HOG
from scripts.utils.conf import Conf
import imutils
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# load the classifier, then initialize the Histogram of Oriented Gradients descriptor
# and the object detector
model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
          cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
w = int(image.shape[1] / 5)
image = imutils.resize(image, width=w)
roi = cv2.resize(image, (250, 440), interpolation=cv2.INTER_AREA)
features = hog.describe(roi).reshape(1, -1)

print(model.predict(features))

cv2.imshow("Image", image)
cv2.waitKey(0)
