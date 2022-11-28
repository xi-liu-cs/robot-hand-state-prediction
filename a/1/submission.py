import pickle as pkl
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import CNN_model

model = CNN_model.CNN().main()

class load(Dataset):
    def __init__(self, path, isTrain=True, transform = None):
        self.transform = transform
        path = path + ('train\\' if isTrain else 'test\\')
        self.pathx = path + 'X\\'
        self.pathy = path + 'Y\\'
        self.data = os.listdir(self.pathx)
        self.isTrain = isTrain
    def __getitem__(self, idx):
        f = self.data[idx]
        img0 = cv2.imread(self.pathx + f + '\\rgb\\0.png')
        img1 = cv2.imread(self.pathx + f + '\\rgb\\1.png')
        img2 = cv2.imread(self.pathx + f + '\\rgb\\2.png')
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        depth = np.load(self.pathx + f + '\\depth.npy')
        field_id = pkl.load(open(self.pathx + f + '\\field_id.pkl', 'rb'))
        if self.isTrain == False:
            return (img0, img1, img2, depth, torch.tensor(int(field_id)))
        y = np.load(self.pathy + f + '.npy')
        return (img0, img1, img2, depth, torch.tensor(int(field_id))), torch.tensor(y)
    def __len__(self):
        return len(self.data)

path_tr = '.\\lazydata\\'
path_te = '.\\lazydata\\'

outfile = 'submission.csv'

output_file = open(outfile, 'w')

titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
         'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']
preds = []

test_data = load(path_te, isTrain = False)
file_ids = test_data[-1]
rgb_data = test_data[0]
model.eval()

for i, data in enumerate(rgb_data):
    # Please remember to modify this loop, input and output based on your model/architecture
    output = model(data[:1, :, :, :].to('cuda'))
    preds.append(output[0].cpu().detach().numpy())

df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame.from_records(preds)], axis = 1, names = titles)
df.columns = titles
df.to_csv(outfile, index = False)
print("Written to csv file {}".format(outfile))