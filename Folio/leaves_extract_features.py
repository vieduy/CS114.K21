# import the necessary packages
from __future__ import print_function
from pyimagesearch.object_detection import helpers
from pyimagesearch.descriptors.hog import HOG
from pyimagesearch.utils import dataset
from pyimagesearch.utils.conf import Conf
from imutils import paths
from scipy import io
import numpy as np
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
print(len(trnPaths))

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
    if 0 <= imageID <= 20:
        labels.append(1)
    elif 21 <= imageID <= 40:
        labels.append(2)
    elif 41 <= imageID <= 60:
        labels.append(3)
    elif 61 <= imageID <= 80:
        labels.append(4)
    elif 81 <= imageID <= 100:
        labels.append(5)
    elif 101 <= imageID <= 120:
        labels.append(6)
    elif 121 <= imageID <= 140:
        labels.append(7)
    elif 141 <= imageID <= 160:
        labels.append(8)
    elif 161 <= imageID <= 180:
        labels.append(9)
    elif 181 <= imageID <= 200:
        labels.append(10)
    elif 201 <= imageID <= 220:
        labels.append(11)
    elif 221 <= imageID <= 240:
        labels.append(12)
    elif 241 <= imageID <= 260:
        labels.append(13)
    elif 261 <= imageID <= 278:
        labels.append(14)
    elif 279 <= imageID <= 298:
        labels.append(15)
    elif 299 <= imageID <= 318:
        labels.append(16)
    elif 319 <= imageID <= 338:
        labels.append(17)
    elif 339 <= imageID <= 358:
        labels.append(18)
    elif 359 <= imageID <= 378:
        labels.append(19)
    elif 379 <= imageID <= 398:
        labels.append(20)
    elif 399 <= imageID <= 418:
        labels.append(21)
    elif 419 <= imageID <= 438:
        labels.append(22)
    elif 439 <= imageID <= 457:
        labels.append(23)
    elif 458 <= imageID <= 477:
        labels.append(24)
    elif 478 <= imageID <= 497:
        labels.append(25)
    elif 498 <= imageID <= 517:
        labels.append(26)
    elif 518 <= imageID <= 537:
        labels.append(27)
    elif 538 <= imageID <= 557:
        labels.append(28)
    elif 558 <= imageID <= 577:
        labels.append(29)
    elif 578 <= imageID <= 597:
        labels.append(30)
    elif 598 <= imageID <= 617:
        labels.append(31)
    elif 618 <= imageID <= 637:
        labels.append(32)
    # update the progress bar
    pbar.update(i)

# grab the distraction image paths and reset the progress bar
pbar.finish()


# dump the dataset to file
pbar.finish()
print("[INFO] dumping features and labels to file...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")