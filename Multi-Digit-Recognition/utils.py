import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, utils, models
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema

from groupcnn import ConvP4, ConvZ2P4, MaxRotationPoolP4, MaxSpatialPoolP4


os.environ['TORCH_HOME'] = '/scratch/arihanth.srikar/'


class MNISTDataset(Dataset):
    def __init__(self, data_name: str='data/mnist_data.npz', label_name: str='data/mnist_labs.npy', num_mutations: int=1, transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.5,), std=(0.5,))])
        else:
            self.transform = transform
        self.y = np.load(label_name)
        self.X = np.load(data_name)
        self.X = self.X.f.arr_0
        for _ in range(num_mutations):
            self.X = np.concatenate((self.X, mutate_images(self.X)))
            self.y = np.concatenate((self.y, self.y))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img, label = self.X[idx], self.y[idx]
        img, label = torch.tensor(img), torch.tensor(label)
        img = self.transform(img)

        return img, label.long()


class NewDataset(Dataset):
    def __init__(self, data_name: str='data/cdata1.npy', label_name: str='data/clab1.npy', isBinary: bool=True, transform=None):
        self.MDR = PreprocessData(data_name, label_name, isBinary)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.5,), std=(0.5,))])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.MDR.X)
    
    def __getitem__(self, idx):
        img, label = self.MDR.X[idx], self.MDR.y[idx]
        img, label = torch.tensor(img), torch.tensor(label)
        img = self.transform(img)

        return img, label.long()


class CustomDataset(Dataset):
    def __init__(self, data_name: str='data/data0.npy', label_name: str='data/lab0.npy', isBinary: bool=True, method: int=2, transform=None):
        self.MDR = PreprocessData(data_name, label_name, isBinary)
        self.method = method
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.5,), std=(0.5,))])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.MDR.X)
    
    def __getitem__(self, idx):
        img, label = self.MDR.X[idx], self.MDR.y[idx]
        
        # segment image to find individual digits
        if self.method == 1:
            digits = self.MDR.segment_to_digits(img)
        elif self.method == 2:
            edges = self.MDR.detect_edges(self.MDR.to_color_image(img), (5,5))
            contours, _, _ = self.MDR.find_contours(self.MDR.to_color_image(img), edges, cv2.CHAIN_APPROX_SIMPLE)
            digits = [self.MDR.gray2binary(self.MDR.resize((self.MDR.extract_contour(img, c)))).squeeze() for c in contours]
            
            if len(digits) != 4:
                digits = rule_based_segmentation(img)
                digits = [self.MDR.resize(x) for x in digits]
                digits = np.array(digits, dtype=np.uint8)
            
            if len(digits) != 4:
                digits = self.MDR.segment_to_digits(img)
            
            if len(digits) != 4:
                del self.MDR.X[idx]
                del self.MDR.y[idx]
                return self.__getitem__(idx)

        # convert to tensor and normalise
        digits, label = torch.tensor(digits), torch.tensor(label)
        digits = self.transform(digits)

        return digits, label


class ConvNet(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class EquiNet(torch.nn.Module):
    def __init__(self, num_classes: int=10):
        super(EquiNet, self).__init__()
        self.conv1 = ConvZ2P4(1, 8, 5)
        self.pool1 = MaxSpatialPoolP4(2)
        self.conv2 = ConvP4(8, 32, 3)
        self.pool2 = MaxSpatialPoolP4(2)
        self.conv3 = ConvP4(32, 64, 3)
        self.pool3 = MaxSpatialPoolP4(2)
        self.conv4 = ConvP4(64, 32, 3)
        self.pool4 = MaxRotationPoolP4()
        self.pool5 = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(32, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(self.pool1(x))
        x = self.conv2(x)
        x = torch.nn.functional.relu(self.pool2(x))
        x = self.conv3(x)
        x = torch.nn.functional.relu(self.pool3(x))
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.pool5(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output


class Resnet50(nn.Module):
    def __init__(self, num_classes: int=10):
        super(Resnet50, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ViT16(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ViT16, self).__init__()
        weights = models.ViT_B_16_Weights.DEFAULT
        self.vit = models.vit_b_16(weights=weights)
        self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.vit.heads = nn.Sequential(
            nn.Linear(768, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.vit(x)
        return x


class PreprocessData():
    def __init__(
        self, 
        data_file_name: str='data/data0.npy', 
        labels_file_name: str='data/lab0.npy',
        isBinary: bool=True,
        ) -> None:
        # Data is stored in the form of numpy arrays
        # These are interpreted as color images by cv2
        self.y = np.load(labels_file_name)
        self.X = np.load(data_file_name)
        self.X = np.array(self.X, dtype=np.uint8)
        if isBinary:
            self.X = self.gray2binary(self.X)

    def gray2binary(self, arr: np.ndarray) -> np.ndarray:
        return np.array([cv2.threshold(x, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] for x in arr])

    def to_color_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    def to_grey_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect_edges(self, image: np.ndarray, blur_intensity: Tuple=(3,3)) -> np.ndarray:
        return cv2.Canny(image=cv2.GaussianBlur(image, blur_intensity, 0), threshold1=100, threshold2=200)
    
    def find_contours(self, image: np.ndarray, edges: np.ndarray, algo: int=cv2.CHAIN_APPROX_NONE) -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
        img_copy = self.gray2binary(image)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, algo)
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        all_countours = []
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            if h < 10 or w < 5:
                continue
            if i and len(c) > 4:
                px, py, pw, ph = cv2.boundingRect(contours[i-1])
                if x - px < 3:
                    continue
            all_countours.append(c)
            cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)
        return all_countours, hierarchy, img_copy

    def extract_contour(self, img: np.ndarray, c: np.ndarray) -> np.ndarray:
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w].copy()

    def resize(self, img: np.ndarray, desired_size: int=28) -> np.ndarray:
        old_size = img.shape[:2]  # (height, width)
        new_size = img.shape[:2]

        if old_size[0] > desired_size or old_size[1] > desired_size:
            ratio = float(desired_size-2)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

        img = cv2.resize(img, (new_size[1], new_size[0]))  # (width, height)

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    def padd_img(self, img: np.array) -> np.array:
        w, h = img.shape
        new_img = np.pad(img, pad_width = ((4,4),(6 ,6)), mode = 'constant')
        new_img = cv2.resize(new_img, (28,28))
        return new_img

    def break_img(self, img: np.array) -> Tuple[np.array, np.array]:
        a= img[:, :14]
        b = img[:, 14:]
        return a, b

    def segment_to_digits(self, img: np.array) -> np.array:
        img_cpy = self.gray2binary(img)
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img_cpy, cv2.CV_32S)
        
        digits, new_digits= [], []
        
        for i in range(0, numLabels):
            if i != 0:
                x = stats[i, cv2.CC_STAT_LEFT] -2
                y = stats[i, cv2.CC_STAT_TOP] - 2 
                w = stats[i, cv2.CC_STAT_WIDTH] + 4
                h = stats[i, cv2.CC_STAT_HEIGHT] + 4
                area = stats[i, cv2.CC_STAT_AREA]
                new_img = img[y:y+h+1, x:x+w+1]
                if area > 20:
                    digits.append(new_img)
                    
        if (len (digits) == 3):
            max_ind = 1
            max_area = 0
            for i in range(len(digits)):
                area = digits[i].shape[0] * digits[i].shape[1]
                if (area > max_area):
                    max_area =area
                    max_ind = i
                    
            digit1, digit2 = self.break_img(self.padd_img(digits[max_ind]))
            new_digits = [self.padd_img(digit1), self.padd_img(digit2)]
            
            for i in range(len(digits)):
                if (i!=max_ind):
                    new_digits.append(self.padd_img(digits[i]))
            digits = new_digits
        else:
            for i in range(len(digits)):
                new_digits.append(self.padd_img(digits[i]))
        
        digits = new_digits
        if len(digits) != 4:
            digits = rule_based_segmentation(img, 10, 20, 4)
            digits = [self.resize(x) for x in digits]

        return np.array(digits, dtype=np.uint8)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def rule_based_segmentation(
    img: np.ndarray,
    min_width: int=10,
    max_width: int=20,
    num_digits: int=4,
    ) -> list:

    imgs = []
    buffer = 5
    small_buffer = 2

    orig_img = img.copy()
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    density_map = np.mean(img.T, axis=1)
    
    seg_pts = [[] for _ in range(num_digits)]
    minimas = signal.find_peaks(1/(smooth(density_map, buffer)+1))[0]
    maximas = signal.find_peaks(smooth(density_map, buffer))[0]

    c1, d_len = 0, len(density_map)
    while c1 < d_len:
        if len(imgs) == num_digits:
            break

        if density_map[c1] != 0:
            # print(c1)
            st = c1
            end = min(c1+max_width, d_len)
            local_mins = minimas[np.logical_and(minimas>st, minimas<=end)]
            
            if len(local_mins) == 0:
                # print('set 1')
                end = min(c1+max_width, d_len)
                try:
                    c1 = minimas[minimas>st][0]
                except:
                    c1 += end
            else:
                for minima in local_mins:
                    found_zeros = np.where(density_map[minima:end] == 0.0)[0]
                    if len(found_zeros):
                        # print('set 2', found_zeros)
                        end = minima+found_zeros[0]-1
                        c1 = minima+found_zeros[-1]-1
                        break
                else:
                    if local_mins[-1]-c1 < 2*buffer-small_buffer:
                        # print('set 3')
                        end = max_width
                        c1 += max_width
                    else:
                        # print('set 4')
                        end = local_mins[-1]+small_buffer
                        c1 = local_mins[-1]+small_buffer
            
            end = max(min_width, end+small_buffer)
            new_img = orig_img[:, st-small_buffer:end+small_buffer].copy()
            h, w = new_img.shape[:2]
            if h > w:
                h1 = (h-w)//2
                h2 = (h-w)-h1
                new_img = np.concatenate((np.zeros((h, h1)), new_img, np.zeros((h, h2))), axis=1)
            imgs.append(new_img)
        c1 += 1

    return imgs

def my_collate(batch):
    """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
    # item: a tuple of (img, label)
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, target]

def skip_collate(batch):
    """Define collate_fn myself because the default_collate_fn throws errors like crazy"""
    # item: a tuple of (img, label)
    data = []
    target = []
    for item in batch:
        d, t = item
        if d.shape[0] == 4:
            data.append(d)
            target.append(t)
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, target]

def mutate_images(images):
    new_images = images.copy()
    _, w = new_images[0].shape

    for i in range(len(new_images)):
        idx = np.random.randint(len(new_images))
        s = np.random.randint(0, w)
        t = np.random.randint(3, 8)

        if s <= w//2:
            new_images[i][:, :t] = images[idx][:, s:s+t].copy()
        else:
            new_images[i][:, -t:] = images[idx][:, s-t:s].copy()
            
    return new_images