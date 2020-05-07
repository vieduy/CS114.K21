# import the necessary packages
from __future__ import print_function
from scripts.descriptors.hog import HOG
from scripts.utils import dataset
from scripts.utils.conf import Conf
from imutils import paths
import progressbar
import argparse
import random
import cv2
import imutils

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the HOG descriptor along with the list of data and labels
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
          cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
data = []
labels = []

# grab the set of ground-truth images and select a percentage of them for training
trnPaths = list(paths.list_images(conf["image_dataset"]))
trnPaths = random.sample(trnPaths, int(len(trnPaths) * conf["percent_gt_images"]))
print("[INFO] describing training ROIs...")

# set up the progress bar
widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()

# loop over the training paths
for (i, trnPath) in enumerate(trnPaths):
    # load the image, convert it to grayscale, and extract the image ID from the path
    image = cv2.imread(trnPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w = int(image.shape[1] / 5)
    image = imutils.resize(image, width=w)
    imageID = trnPath[trnPath.rfind("_") + 1:].replace(".jpg", "")
    imageID = int(imageID)

    roi = cv2.resize(image, (250, 440), interpolation=cv2.INTER_AREA)
    features = hog.describe(roi)
    data.append(features)
    labels.append(int(imageID/20) + 1)

    # update the progress bar
    pbar.update(i)

# grab the distraction image paths and reset the progress bar
pbar.finish()


# dump the dataset to file
pbar.finish()
print("[INFO] dumping features and labels to file...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")