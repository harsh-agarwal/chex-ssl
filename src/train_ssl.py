import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
# from dataset_wrapper import label_mapping
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../data/', help='path to the folder where prepared data is downloaded')
parser.add_argument('--cat', type=int, default=1, help='category to train for')
parser.add_argument('--num_label_sam', type=int, default=2000, help='fraction of labeled data')
parser.add_argument('--num_unlabel_sam', type=int, default=20000, help='fraction of labeled data')
parser.add_argument('--nc', type=int, default=2500, help='number of components for PCA')
parser.add_argument('--kernel', type=str, default='rbf', help='which kernel to use: {rbf, knn}')
parser.add_argument('--num_iter', type=int, default=30, help='number of iterations for label spreading')
parser.add_argument('--num_neighbors', type=int, default=7, help='number of nearest neighbors for knn kernel')
parser.add_argument('--gamma', type=float, default=20, help='gamma value for label spreading')
parser.add_argument('--exp_name', type=str, default='ssl_rbf', help='experiment name')
parser.add_argument('--save_path', type=str, default='../outputs/', help='save path')
opt = parser.parse_args()

num_label_sam = opt.num_label_sam
num_unlabel_sam = opt.num_unlabel_sam

if opt.kernel=='rbf':
    exp_name = '{}_rbf_frac{}_nc{}_iter{}_gamma{}'.format(opt.exp_name, opt.num_label_sam, opt.nc, opt.num_iter, opt.gamma)
else:
    exp_name = '{}_knn_frac{}_nc{}_nn{}'.format(opt.exp_name, opt.num_label_sam, opt.num_neighbors, opt.nc, opt.num_iter)

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)
if not os.path.exists(os.path.join(opt.save_path, exp_name)):
    os.makedirs(os.path.join(opt.save_path, exp_name))

print('Starting : ', exp_name)
print('='*50)
for arg in vars(opt):
    print('{} : {}'.format(arg, getattr(opt, arg)))
print('='*50)

# #############################################################################
# Data loading and preparation
# #############################################################################
X_train = np.load(os.path.join(opt.data_path, 'train_input.npy'))
Y_train = np.load(os.path.join(opt.data_path, 'train_output.npy'))
X_test = np.load(os.path.join(opt.data_path, 'test_input.npy'))
Y_test = np.load(os.path.join(opt.data_path, 'test_output.npy'))


train_input_labelled = X_train[:num_label_sam,:]
train_out_labelled = Y_train[:num_label_sam,(opt.cat - 1)]
train_input_unlabelled = X_train[num_label_sam:(num_unlabel_sam + num_label_sam),:]
train_out_unlabelled = -1 * np.ones((num_unlabel_sam), dtype = int)
Y_test = Y_test[:,opt.cat - 1]
# would be used for transductive metrics! 
orig_label_train = Y_train[:(num_unlabel_sam+num_label_sam),opt.cat-1]
# ipdb.set_trace()
train_in = np.concatenate((train_input_labelled, train_input_unlabelled), axis = 0)
train_out = np.concatenate((train_out_labelled, train_out_unlabelled), axis = 0)
 


# class_names = []
# for k,v in sorted(label_mapping.items(), key=lambda kv:kv[1]):
#     class_names.append(v)

# X = np.concatenate((X_train, X_test))
# Y = np.concatenate((Y_train, Y_test))
# images = np.copy(X_train)
# images = np.reshape(images, (-1, 64, 64))

# shuffle everything around
# num_total = len(Y_train)
# rng = np.random.RandomState(0)
# indices = np.arange(num_total)
# rng.shuffle(indices)

# select fraction of data to be labelled
# num_labeled = int(opt.fraction * len(X_train))
# unlabeled_set = indices[num_labeled:]
# Y_ssl = np.copy(Y_train)
# Y_ssl[unlabeled_set] = -1

print('Applying PCA to select {} components.'.format(opt.nc))
pca = PCA(n_components=opt.nc)
pca_transformer = pca.fit(train_in)
X_train = pca_transformer.transform(train_in)
X_test = pca_transformer.transform(X_test)

# #############################################################################
# Learn with LabelSpreading
print('Fitting the model now.')
if opt.kernel=='rbf':
    lp_model = label_propagation.LabelSpreading(gamma=opt.gamma, max_iter=opt.num_iter, n_jobs=10)
else:
    lp_model = label_propagation.LabelSpreading(kernel='knn', n_neighbors=opt.num_neighbors, max_iter=opt.num_iter, n_jobs=10)

lp_model.fit(X_train, train_out)

# if opt.fraction < 1:
#     predicted_labels = lp_model.transduction_[unlabeled_set]
#     true_labels = Y_train[unlabeled_set]
#     cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

#     print('Label Spreading model: %d labeled & %d unlabeled points (%d total)' %
#           (num_labeled, num_total - num_labeled, num_total))

#     print(classification_report(true_labels, predicted_labels))

#     # print('Confusion matrix: ')
#     # print(cm)

#     # #############################################################################
#     # Calculate uncertainty values for each transduced distribution
#     pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

#     # #############################################################################
#     # Pick the top 10 most uncertain labels
#     uncertainty_index = np.argsort(pred_entropies)[-10:]

#     # #############################################################################
#     # Plot
#     f = plt.figure(figsize=(7, 5))
#     for index, image_index in enumerate(uncertainty_index):
#         image = images[image_index]

#         sub = f.add_subplot(2, 5, index + 1)
#         sub.imshow(image, cmap=plt.cm.gray_r)
#         plt.xticks([])
#         plt.yticks([])
#         sub.set_title('predict: %i\ntrue: %i' % (
#             lp_model.transduction_[image_index], Y_train[image_index]))

#     f.suptitle('Learning with small amount of labeled data')
#     # plt.show()
#     plt.savefig('{}/uncertain_labels_transductive.png'.format(os.path.join(opt.save_path, exp_name)))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # ipdb.set_trace()
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print('Normalized confusion matrix:')
    # else:
    #     print('Confusion matrix, without normalization:')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    return ax

# #############################################################################
# make transductive confusion matrix plots
# #############################################################################
# plot_confusion_matrix(true_labels, predicted_labels, classes=class_names, normalize=True,
#                       title='Confusion matrix, with normalization')
# plt.savefig('{}/transductive_cf_normalized.png'.format(os.path.join(opt.save_path, exp_name)))
# plt.close()
# plot_confusion_matrix(true_labels, predicted_labels, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.savefig('{}/transductive_cf_unnormalized.png'.format(os.path.join(opt.save_path, exp_name)))
# plt.close()

# #############################################################################
# make inductive inference on the test data
# #############################################################################
predicted_labels = lp_model.predict(X_test)
# plot_confusion_matrix(Y_test, predicted_labels, classes=class_names, normalize=True,
#                      title='Confusion matrix, with normalization')
# plt.savefig('{}/inductive_cf_normalized.png'.format(os.path.join(opt.save_path, exp_name)))
# plt.close()
# plot_confusion_matrix(Y_test, predicted_labels, classes=class_names,
#                      title='Confusion matrix, without normalization')
# plt.savefig('{}/inductive_cf_unnormalized.png'.format(os.path.join(opt.save_path, exp_name)))
# plt.close()
print("Test data confison matrix:")
print(confusion_matrix(Y_test, predicted_labels))

print('-'*50)
print('Mean accuracy  on the given test data and labels: {}'.format(lp_model.score(X_test, Y_test)))
print('-'*50)
