import numpy as np
from tqdm import tqdm

from utils import my_collate, skip_collate
from utils import NewDataset, CustomDataset, MNISTDataset
from utils import ConvNet, EquiNet, Resnet50

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BEST_PARAMS = {
    'BATCH_SIZE': 16,
    'LEARNING_RATE': 0.001,
    'NUM_WORKERS': 2,
    'EPOCHS': 20,
}

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
NUM_WORKERS = 2

transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5,), std=(0.5,))])

# mnist_train_data = datasets.MNIST('./data', download=True, train=True, transform=transform)
# mnist_val_data = datasets.MNIST('./data', download=True, train=False, transform=transform)

mnist_data = MNISTDataset('data/mnist_data.npz', 'data/mnist_labs.npy', 3)
train_size = int(0.8 * len(mnist_data))
val_size = len(mnist_data) - train_size
mnist_train_data, mnist_val_data = torch.utils.data.random_split(mnist_data, [train_size, val_size])

mnist_train_loader = DataLoader(mnist_train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
mnist_val_loader = DataLoader(mnist_val_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

dataset = NewDataset('data/cdata.npy', 'data/clab.npy')
train_size = int(0.4 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

custom_train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=my_collate, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
custom_test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=my_collate, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
custom_val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=my_collate, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


def train_and_validate(
    model_args: list,
    file_name: str,
    num_epoch: int=10):

    model, train_loader, val_loader, optimizer, criterion, scheduler = model_args

    best_loss = np.Inf
    count = 0
    last_lr = scheduler.optimizer.param_groups[0]['lr']
    
    model.train()
    for epoch in range(1, num_epoch+1):
        correct, total = 0, 0
        losses = []

        for _, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            losses.append(loss.data.detach().cpu())

            loss.backward()
            optimizer.step()

        print(f'\nTrain loss {sum(losses)/total:.5f} accuracy: {100*correct/total:2.2f}')

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            losses = []
            
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                losses.append(loss.data.detach().cpu())

            curr_loss = sum(losses)/total
            scheduler.step(curr_loss)
            lr = scheduler.optimizer.param_groups[0]['lr']

            print(f"Evaluation loss {curr_loss:.5f} accuracy: {100*correct/total:2.2f}, lr {lr}")
            if lr <= 1e-5:
                print('Learning rate dropped too low to be able to learn anything')
                return

            if curr_loss < best_loss:
                last_lr = lr
                best_loss = curr_loss
                print(f'Last saved on epoch {epoch}\n')
                torch.save(model.state_dict(), file_name)
            elif last_lr != lr:
                count = 0
            else:
                count += 1
                if count >= 5:
                    model.load_state_dict(torch.load(file_name))
                    optimizer = optim.AdamW(params=model.parameters(), lr=last_lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

def test(
    model,
    criterion,
    test_loader):

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        losses = []
        
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            losses.append(loss.data.detach().cpu())

        print(f'Test loss {sum(losses)/total:.5f} accuracy: {100*correct/total:2.2f}')

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

        print(f'Accuracy: {100*correct/total:2.2f}%')
    
    return new_imgs, new_labels

LEARNING_RATE = 1e-3
EPOCHS = 25

# models = [ConvNet(10), EquiNet(10), Resnet50(10)]
# models_weights = ['models/ConvNetWeights.ckpt', 'models/EquiNetWeights.ckpt', 'models/R50Weights.ckpt']
models = [EquiNet(10), ConvNet(10)]
models_weights = ['models/EquiNetWeights.ckpt', 'models/ConvNetWeights.ckpt']

for model, model_weights in zip(models, models_weights):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    print(f'\nTraining {model_weights}')
    
    model_args = model, mnist_train_loader, mnist_val_loader, optimizer, criterion, scheduler
    train_and_validate(model_args, model_weights, EPOCHS)

    model.load_state_dict(torch.load(model_weights))
    test(model, criterion, custom_test_loader)

    print(f'\nEvaluating on {model_weights}')
    for i in range(3):
        dataset = CustomDataset(f'data/data{i}.npy', f'data/lab{i}.npy', True, 1)
        data_loader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=skip_collate)

        new_images, new_labels = test_with_batching(model, data_loader)

    optimizer = optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    model_args = model, custom_train_loader, custom_val_loader, optimizer, criterion, scheduler
    train_and_validate(model_args, model_weights, EPOCHS)

    model.load_state_dict(torch.load(model_weights))
    test(model, criterion, custom_test_loader)

    print(f'\nEvaluating on {model_weights}')
    for i in range(3):
        dataset = CustomDataset(f'data/data{i}.npy', f'data/lab{i}.npy', True, 1)
        data_loader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=skip_collate)

        new_images, new_labels = test_with_batching(model, data_loader)
