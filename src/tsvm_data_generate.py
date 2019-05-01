import numpy as np
from sklearn.svm import SVC
import random
from sklearn.decomposition import PCA
import argparse
X_train = np.load('../data/train_input.npy')
y_train = np.load('../data/train_output.npy')

X_test = np.load('../data/test_input.npy')
y_test = np.load('../data/test_output.npy')

unlabeled_frac = 0.9
total_samples = 10000

print('Sampling')
# samples = random.sample(list(range(y_train.shape[0])),22000)
sampled_X_train = X_train[:total_samples]
# Only extract 0th label
split_idx =int(unlabeled_frac*sampled_X_train.shape[0])
labeled_sampled_X_train = sampled_X_train[split_idx:]

print('PCA')
pca = PCA(n_components=1000)
pca_transformer = pca.fit(sampled_X_train)
data_red_train = pca_transformer.transform(sampled_X_train)

num_classes = y_train.shape[1]
for class_idx in range(num_classes):
    print('Removing labels for class type:', class_idx)
    sample_y_train = y_train[:total_samples, class_idx]
    sample_y_train[np.where(sample_y_train == 0)] = -1
    sample_y_train[0:split_idx] = 0

    data_file = open('new_chex_train_'+str(class_idx)+'.txt', 'w')

    # print(sample_y_train)
    num_samples = sample_y_train.shape[0]
    num_features = data_red_train.shape[1]
    for y_idx in range(num_samples):
        feature_list = [str(int(sample_y_train[y_idx]))]
        for x_idx in range(num_features):
            feature_list.append(str(x_idx+1)+':'+str(data_red_train[y_idx, x_idx]))
        # First and last chars are [ ]
        feature_string = ' '.join(feature_list)
        data_file.write(feature_string+'\n')
        # print(feature_string)
        feature_list = []
    data_file.close()

    data_red_test = pca_transformer.transform(X_test)
    y_test_n = y_test[:, class_idx]

    data_file = open('new_chex_test_'+str(class_idx)+'.txt', 'w')

    num_samples = y_test_n.shape[0]
    num_features = data_red_test.shape[1]
    for y_idx in range(num_samples):
        feature_list = [str(int(y_test_n[y_idx]))]
        for x_idx in range(num_features):
            feature_list.append(str(x_idx+1)+':'+str(data_red_test[y_idx, x_idx]))
        # First and last chars are [ ]
        feature_string = ' '.join(feature_list)
        data_file.write(feature_string+'\n')
        # print(feature_string)
        feature_list = []
    data_file.close()
