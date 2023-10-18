import numpy as np
import os
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
N = 24 # 24 trials

# Prepare X_train, y_train, X_test, y_test from dataset
perSample = ['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS']
directories = ["../data/eeg_feature_smooth/{}/".format(i+1) for i in range(3)]
 
channel_coords = [
    ['0', '0', 'AF3', 'FP1', 'FPZ', 'FP2', 'AF4', '0', '0'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['0', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '0'],
    ['0', '0', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '0', '0']
]

channel_list = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
    'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
    'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2',
    'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
    'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
    'O1', 'OZ', 'O2', 'CB2'
]

coord_dict = {}
for n in range(len(channel_list)):
    for i, l in enumerate(channel_coords):
        for j, x in enumerate(l):
            if (channel_list[n] == x):
                coord_dict[n] = (i,j)

array = np.zeros(shape=(3,15,24,4,8,9,5,64))
for h, dire in enumerate(directories):
    data = [loadmat(dire + file) for file in os.listdir(dire)]
    for i, bigsample in enumerate(data):
        for j in range(24):
            for k, key in enumerate(perSample):
                sample = np.transpose(np.array(bigsample[key + str(j+1)]), (0,2,1))
                sample = np.pad(sample, [(0,0), (0,0), (0, 64-sample.shape[2])], mode = 'constant')
                for l, channel in enumerate(sample):
                    array[h][i][j][k][coord_dict[l][0]][coord_dict[l][1]] = channel

X = array.reshape(np.prod(array.shape[0:3]), *array.shape[3:]) # X will have shape (3*15*24, 4,8,9,5,64) = (1080,4,8,9,5,64)
X = X.transpose(0, 5, 1,2,3,4) # (1080,64,4,8,9,5)
X = X.reshape(X.shape[0], X.shape[1], np.prod(X.shape[2:])) # (1080,64,1440)

# Process eye
directories = ["../data/eye_feature_smooth/{}/".format(i+1) for i in range(3)]
array_eye = np.zeros(shape=(len(directories),len(os.listdir(directories[0])), N, 31, 64)) # feature : 31 eyes

for h, dire in enumerate(directories):
    data = [loadmat(dire + file) for file in os.listdir(dire)]
    for i, bigsample in enumerate(data):
        sample = bigsample["eye_" + str(i+1)]
        sample = np.pad(sample, [(0,0), (0,64-sample.shape[1])], mode='constant')
        array_eye[h] = sample
X_eye = array_eye.reshape(np.prod(array_eye.shape[0:3]), *array_eye.shape[3:])
X_eye = X_eye.transpose(0,2,1)

final_x = np.concatenate((X, X_eye), axis=2) # Concat into a single array input

# Create y
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
y = np.array(session1_label * 15 + session2_label * 15 + session3_label * 15) 

# Train test split
X_train, X_test, y_train, y_test = train_test_split(final_x, y, test_size=0.2, random_state=42)