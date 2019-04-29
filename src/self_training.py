from joblib import dump, load
from sklearn.svm import SVC
import random
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import argparse
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type = str, default = '../models/classifier.joblib', help = 'the path of the classifer to talke for pre training')
parser.add_argument('--cat', type=int, default='1', help='category to train the svm for')
parser.add_argument('--num_sam', type = int, default = 10000, help='num_samples_to_use')
parser.add_argument('--pca_com', type = int, default = 1000, help='number of pca compnents to reduce to')
parser.add_argument('--C', type = int, default = 1, help='C value SVM hyper parameter')
parser.add_argument('--gamma', type = str, default = 'auto', help ='parameter for svm')
parser.add_argument('--save', type = bool, default = True, help = 'a boolean for allowing to save or not')
parser.add_argument('--save_path', type = str, default = '../models/classifier_self_trained.joblib', help = 'path for saving the models')
parser.add_argument('--max_epochs', type = int, default = 5, help = 'number of epochs that you want the svm to train for!')
opt = parser.parse_args()

model_path = opt.model_path
cat = opt.cat
num_sam = opt.num_sam
pca_com = opt.pca_com
C = opt.C
gamma = opt.gamma
save = opt.save
save_path = opt.save_path 
max_epochs = opt.max_epochs

print('-'*50)

print('Using the base classifier at ' + model_path)
print('category being trained:' + str(cat))
print('Number of unlabelled samples that we are using are: ' + str(num_sam))
print('C : ' + str(C))
print('gamma : ' + str(gamma))

clf = load(model_path)

orig_input_data = np.load("../data/train_input.npy")
orig_output_data = np.load("../data/train_output.npy")

input_data = orig_input_data[10000:10000 + num_sam,:]
# would be using the ground truth for understanding the transductive statistics! 
output_data = orig_output_data[10000:10000 + num_sam,(cat - 1)]

print("Working on the pca -  reducing the dimesnionality")
pca = PCA(n_components=pca_com)
pca_transformer = pca.fit(input_data)
data_red_train = pca_transformer.transform(input_data)

for epoch in range(0, max_epochs):
    print("creating the output usign the trained classifier!")
    pred_label_train = clf.predict(data_red_train)
    # train it using these labels
    clf.fit(data_red_train,pred_label_train)

# get the transductive stats 
print("Train data (trasnductive!) confusion matrix")
pred_label_train = clf.predict(data_red_train)
print(confusion_matrix(output_data, pred_label_train))

# get the stats on the test set 
data_test = np.load("../data/test_input.npy") 
test_label_gt = np.load("../data/test_output.npy")
test_output = test_label_gt[:,cat - 1]

# apply pca transformation on the test data
test_data_red = pca_transformer.transform(data_test)
pred_label = clf.predict(test_data_red)

# print the confusion matrices!
print("Test data confison matrix:")
print(confusion_matrix(test_output, pred_label))

if(opt.save==True):
    print("saving the model")
    dump(clf,opt.save_path)
















