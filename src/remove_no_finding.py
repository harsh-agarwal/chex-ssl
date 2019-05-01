import numpy as np 

test_in_red = np.load('../data/test_in_red.npy')
test_out = np.load('../data/test_output.npy')

no_find = test_out[:,10]

idx_keep = np.where(no_find!=1)

test_in_red = test_in_red[idx_keep[0]]
test_out_removed = test_out[idx_keep[0]]

np.save('../data/test_in_red_removed.npy', test_in_red)
np.save('../data/test_out_red_removed.npy', test_out_removed)