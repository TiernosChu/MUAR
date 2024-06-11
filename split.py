import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = np.load('./SWellEx-96-S5/data.npy')
label = pd.read_csv('./SWellEx-96-S5/label.csv')

train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.125)

np.save('./SWellEx-96-S5/train_data_1.npy', train_data)
train_label.to_csv('./SWellEx-96-S5/train_label_1.csv', index=False)
np.save('./SWellEx-96-S5/valid_data_1.npy', valid_data)
valid_label.to_csv('./SWellEx-96-S5/valid_label_1.csv', index=False)
np.save('./SWellEx-96-S5/test_data_1.npy', test_data)
test_label.to_csv('./SWellEx-96-S5/test_label_1.csv', index=False)
