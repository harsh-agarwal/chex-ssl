import subprocess
import glob
for dataset in glob.glob('chex_train*'):
    command = ['../models/svm_learn -z c -c 10 -t 2 -g 0.001', dataset, dataset.split('.')[0]+'_model']
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
