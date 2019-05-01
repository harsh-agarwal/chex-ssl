from sklearn.svm import SVC
import random
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

parser = argparse.ArgumentParser()
parser.add_argument('--cat', type=int, default='1', help='category to train the svm for')
parser.add_argument('--num_sam', type = int, default = 5000, help='num_samples_to_use')
parser.add_argument('--pca_com', type = int, default = 1000, help='number of pca compnents to reduce to')
parser.add_argument('--C', type = float, default = 1, help='C value SVM hyper parameter')
parser.add_argument('--gamma', type = float, default = 1e-3, help ='parameter for svm')
parser.add_argument('--save', type = bool, default = True, help = 'a boolean for allowing to save or not')
parser.add_argument('--save_path', type = str, default = '../models/classifier.joblib', help = 'path for saving the models')
opt = parser.parse_args()

print('-'*50)
print("Hyperparameters: ")
print("C = " + str(opt.C))
print("category = " + str(opt.cat))
print("number of training samples labelled = " + str(opt.num_sam))
print("number of pca components = " + str(opt.pca_com))
print("gamma = " + str(opt.gamma))
print('-'*50)

category = opt.cat
num_samples_train = opt.num_sam
pca_components = opt.pca_com
C_value = opt.C
gamma = opt.gamma

orig_input_data = np.load("../data/train_input.npy")
orig_output_data = np.load("../data/train_output.npy")

input_data = orig_input_data[:num_samples_train,:]
output_data = orig_output_data[:num_samples_train,(category - 1)]

print("Working on the pca -  reducing the dimesnionality")
pca = PCA(n_components=pca_components)
pca_transformer = pca.fit(input_data)
data_red_train = pca_transformer.transform(input_data)

clf = SVC(C = C_value, gamma = gamma)
clf.fit(data_red_train,output_data)

pred_label_train = clf.predict(data_red_train)

data_test = np.load("../data/test_input.npy") 

test_label_gt = np.load("../data/test_output.npy")
test_output = test_label_gt[:,category - 1]


# apply pca transformation on the test data
test_data_red = pca_transformer.transform(data_test)
pred_label = clf.predict(test_data_red)

print("Train data confusion matrix")
print(confusion_matrix(output_data, pred_label_train))

print("Test data confison matrix:")
print(confusion_matrix(test_output, pred_label))
print(classification_report(test_output, pred_label))

if(opt.save==True):
    print("saving the model")
    dump(clf,opt.save_path)



# TODO: code for AUC under the ROC to be written
# may be we can write a seperate function that does that, given the true labels and the predicted labels 











