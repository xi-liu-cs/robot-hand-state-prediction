import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle as pkl
from torch import Tensor
from joblib import dump, load

class Data_Preprocessing:
    def __init__(self):
        self.message = "Final Project of Xi Liu"

    def tensorToArray(self, data, isTrain=True):
        n_samples = len(data)
        if isTrain:
            (img0_, img1_, img2_, depth_, field_id_), y_ = data[0]
            img_shape, img1_shape, img2_shape, depth_shape, n_y = img0_.shape, img1_.shape, img2_.shape, depth_.shape, len(y_)
            y_array = np.empty(shape=(n_samples, n_y))
        else:
            (img0_, img1_, img2_, depth_, field_id_) = data[0]
            img_shape, img1_shape, img2_shape, depth_shape = img0_.shape, img1_.shape, img2_.shape, depth_.shape
            field_id_array = np.empty(shape=(n_samples, 1))
        img0_array = np.empty(shape=(n_samples, img_shape[0], img_shape[1], img_shape[2]))
        img1_array = np.empty(shape=(n_samples, img1_shape[0], img1_shape[1], img1_shape[2]))
        img2_array = np.empty(shape=(n_samples, img2_shape[0], img2_shape[1], img2_shape[2]))
        depth_array = np.empty(shape=(n_samples, depth_shape[0], depth_shape[1], depth_shape[2]))

        for inx, d in enumerate(data):
            # print(inx)
            if isTrain:
                (img0, img1, img2, depth, field_id), y = d
                y_array[inx, :] = np.array(y)
                img0_array[inx, :, :, :] = img0
                img1_array[inx, :, :, :] = img1
                img2_array[inx, :, :, :] = img2
                depth_array[inx, :, :, :] = depth
            else:
                (img0, img1, img2, depth, field_id) = d
                field_id_array[inx] = field_id.numpy()
                img0_array[inx, :, :, :] = img0
                img1_array[inx, :, :, :] = img1
                img2_array[inx, :, :, :] = img2
                depth_array[inx, :, :, :] = depth
        if isTrain:
            return img0_array, img1_array, img2_array, depth_array, y_array
        return img0_array, img1_array, img2_array, depth_array, field_id_array

    # The input shape should be (samples, width, height, 1)
    def depth_normalization(self, depth):
        # normalized data = (data - Min number) / (Max number - Min number)
        """min_num = np.min(depth)
        max_num = np.max(depth)
        normalized_depth = (depth - min_num) / (max_num - min_num)
        return normalized_depth"""
        return depth / 1000.0

    # The input shape should be (samples, width, height, 1)
    def img_normalization(self, img):
        # Original image data is from 0-255, and we want to scale data to 0-1. Thus, we can just divide original data by 255.
        normalized_img = img / 255.0
        return normalized_img

    def combine_image_depth(self, img, depth, whichImg=0):
        new_img = np.empty(shape=(img.shape[0], img.shape[1], img.shape[2], img.shape[3] + 1))

        # First, try only use one image (img0), so use the first depth only.
        depth0 = depth[:, whichImg, :, :]

        for inx, _ in enumerate(img):
            # 2D array (224, 224) to 3D array (224, 224, 1)
            depth_4d = np.expand_dims(depth0[inx], 2)
            # combine img and depth into one array
            new_img[inx] = np.concatenate((img[inx], depth_4d), axis=2)

        return new_img

    # Expected input data size for CNN is (Samples, Channels, Heights, Widths).
    # Thus, we have to reshape original data size (Samples, Heights, Widths, Channels) to the new size mentioned above.
    def reshape_data(self, data):
        samples = data.shape[0]
        channels = data.shape[3]
        heights = data.shape[1]
        widths = data.shape[2]
        new_data = np.empty(shape=(samples, channels, heights, widths))

        for i in range(channels):
            new_data[:, i, :, :] = data[:, :, :, i]
        return new_data

"""class load_images(Dataset):
    def __init__(self, path, isTrain=True, transform=None):
        self.transform = transform
        self.path = path + ('train\\' if isTrain else 'test\\')
        self.pathx = self.path + 'X\\'
        self.pathy = self.path + 'Y\\'
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
        return len(self.data)"""

class load_images(Dataset):
    def __init__(self, path, isTrain=True, transform=None):
        self.transform = transform
        self.path = path + ('train/' if isTrain else 'test/')
        self.pathx = self.path + 'X/'
        self.pathy = self.path + 'Y/'
        self.data = os.listdir(self.pathx)
        self.isTrain = isTrain

    def __getitem__(self, idx):
        f = self.data[idx]
        img0 = cv2.imread(self.pathx + f + '/rgb/0.png')
        img1 = cv2.imread(self.pathx + f + '/rgb/1.png')
        img2 = cv2.imread(self.pathx + f + '/rgb/2.png')
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        depth = np.load(self.pathx + f + '/depth.npy')
        field_id = pkl.load(open(self.pathx + f + '/field_id.pkl', 'rb'))
        if self.isTrain == False:
            return (img0, img1, img2, depth, torch.tensor(int(field_id)))
        y = np.load(self.pathy + f + '.npy')
        return (img0, img1, img2, depth, torch.tensor(int(field_id))), torch.tensor(y)

    def __len__(self):
        return len(self.data)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

class CNN:
    def __init__(self):
        self.message = "Final Project of Xi Liu"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def ResNet50(self, num_classes, channels=3):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)

    def ResNet101(self, num_classes, channels=3):
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)

    def ResNet152(self, num_classes, channels=3):
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)

    def train_model(self, dataloader, model, loss_fn, optimizer, n_epoch):
        for e in range(n_epoch):
            model.train()
            # Print epoch
            print(f'Starting epoch {e + 1}')
            # Set current loss value
            current_loss = 0.0
            for batch_inx, data in enumerate(dataloader): # for batch_inx, (x, y) in enumerate(dataloader):
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                # pred *= 1000.0
                y *= 1000.0
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if batch_inx % 10 == 0:
                    print('Loss after mini-batch %5d: %.3f' % (batch_inx + 1, current_loss / 10))
                    current_loss = 0.0
        return model

    def main(self, loadname, pre_trained_model = None):
        if pre_trained_model == None:
            model = self.ResNet50(num_classes = 12, channels = 4)
        else:
            model = pre_trained_model
        model = model.to(self.device)

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # read_prep_data = load('lx_preprocessed_data.joblib')
        read_prep_data = load(loadname)
        tr_data, tr_y = read_prep_data[0], read_prep_data[1]

        dataset_train = TensorDataset(Tensor(tr_data), Tensor(tr_y))
        train_dataloader = DataLoader(dataset=dataset_train, batch_size=8, shuffle=True, num_workers=2) # batchsize = 64

        # model = self.train_model(dataloader=train_dataloader, model=model, loss_fn=loss_function, optimizer=optimizer, n_epoch=30)
        model = self.train_model(dataloader=train_dataloader, model=model, loss_fn=loss_function, optimizer=optimizer, n_epoch=30)

        return model

    def pred(self, model_name, loadname):
        model = torch.jit.load(model_name)
        model.eval()

        # read_prep_data = load('lx_preprocessed_data.joblib')
        read_prep_data = load(loadname)
        tr_data, tr_y, te_data = read_prep_data[0], read_prep_data[1], read_prep_data[2]

        predictions = np.zeros(shape=(te_data.shape[0], 12))
        dataset_test = TensorDataset(Tensor(te_data), Tensor(predictions))
        batch_size = 16
        test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

        for batch_inx, (x, y) in enumerate(test_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            test_pred = model(x)
            predictions[batch_inx*batch_size:(batch_inx+1)*batch_size, :] = test_pred.detach().cpu().numpy()
            if batch_inx % 10 == 0:
                print('Batch Index: ' + str(batch_inx))
        return predictions

