# import the necessary packages
from __future__ import print_function

from sklearn.metrics import accuracy_score

from scripts.utils import dataset
from scripts.utils.conf import Conf
from sklearn.svm import SVC
import numpy as np
import argparse
import pickle
from sklearn.model_selection import train_test_split

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file and the initial dataset
print("[INFO] loading dataset...")
conf = Conf(args["conf"])
(data, labels) = dataset.load_dataset(conf["features_path"], "features")

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=130)


# train the classifier
print("[INFO] training classifier...")
model = SVC(kernel="linear", C=conf["C"], probability=False, random_state=42)
model.fit(x_train, y_train)


# dump the classifier to file
print("[INFO] dumping classifier...")
f = open(conf["classifier_path"], "wb")
f.write(pickle.dumps(model))
f.close()

y_pred = model.predict(x_test)
print("Accuracy of SVM: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))
