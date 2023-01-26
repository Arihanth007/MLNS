import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, utils, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import cv2
import numpy as np
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema

from utils import PreprocessData, NewDataset
from utils import smooth, rule_based_segmentation, skip_collate
from utils import ConvNet, CustomDataset, EquiNet, Resnet50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_with_batching(model, data_loader):
    model.eval()
    
    with torch.no_grad():
        correct, total = 0, 0
        new_imgs, new_labels = [], []
        
        for images, labels in tqdm(data_loader):
            _, _, x, y = images.shape
            images = images.view(-1, x, y).unsqueeze(dim=1)
            images = images.float().to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            predicted = predicted.view(-1, 4)
            digits_pred = predicted.detach().cpu().numpy()
            predicted = predicted.sum(dim=1)
            
            num_correct = (predicted == labels).sum().item()
            correct += num_correct
            
            if num_correct:
                for new_img, new_label in zip(images.detach().cpu().numpy().squeeze(), digits_pred):
                    new_imgs.append(new_img)
                    new_labels.append(new_label)

        print(f'Accuracy: {100 * correct / total} %')
    
    return new_imgs, new_labels

def test_on_custom_data(model, data_loader):
    model.eval()
    
    with torch.no_grad():
        correct, total = 0, 0
        new_imgs, new_labels = [], []
        
        for images, labels in tqdm(data_loader):
            images = images.permute(1,0,2,3).float().to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            digits_pred = predicted.detach().cpu().numpy()
            predicted = predicted.sum().unsqueeze(dim=0)
            
            num_correct = (predicted == labels).sum().item()
            correct += num_correct
            
            if num_correct:
                for new_img, new_label in zip(images.detach().cpu().numpy().squeeze(), digits_pred):
                    new_imgs.append(new_img)
                    new_labels.append(new_label)

        print(f'Accuracy: {100 * correct / total} %')
    
    return new_imgs, new_labels


BATCH_SIZE = 100
NUM_WORKERS = 2

models = [ConvNet(10), EquiNet(10), Resnet50(10)]
models_weights = ['models/ConvNetWeights.ckpt', 'models/EquiNetWeights.ckpt', 'models/R50Weights.ckpt']

for model, model_weights in zip(models, models_weights):
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_weights))

    print(f'\nEvaluating on {model_weights}')
    for i in range(3):
        dataset = CustomDataset(f'data/data{i}.npy', f'data/lab{i}.npy', True, 1)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=skip_collate)

        new_images, new_labels = test_with_batching(model, data_loader)
        # np.save(f'data/ndata{i}', new_images)
        # np.save(f'data/nlab{i}', new_labels)
    print()
