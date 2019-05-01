import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_test = np.load('../data/test_output.npy')
for label_class in range(15):
    print('TSVM: Label', label_class)
    test_pred = np.loadtxt('../data/test_predictions_'+str(label_class))
    test_pred[test_pred > 0] = 1
    test_pred[test_pred < 0] = 0

    y_test_class = y_test[:,label_class]

    print(classification_report(test_pred, y_test_class))
