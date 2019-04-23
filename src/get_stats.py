from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np
import ipdb 

# image_indices, labels, split_index = read_dataset(csv_dataset_path='misc/Data_Entry_2017.csv', train_test_split=0.9)
input_array = np.load('../data/dataset_All/input_array.npy')
output_array = np.load('../data/dataset_all/output_array.npy')

data = np.concatenate((input_array,output_array), axis = 1)

np.random.shuffle(data)

num_samples = data.shape[0]

num_samples_train = int(0.8 * num_samples)

train = data[:num_samples_train,:]
test = data[(num_samples_train + 1):,:]

train_input = train[:,:-15]
train_output = train[:,-15:]

test_input = test[:,:-15]
test_output = test[:,-15:]

num_elem_each_class = np.sum(train_output, axis = 0)

