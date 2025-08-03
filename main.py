'''Train CIFAR10 with PyTorch.'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('/blue/azare/samgallic/Research/pytorch-cifar/logger.py')))

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import random
import numpy as np
from collections import Counter

import torchvision
import torchvision.transforms as transforms

from dotenv import load_dotenv
import argparse

from models import mnn_resnet, ResNet18
from utils import progress_bar
from heatmap import plot_heatmap
import logger

import json
from PIL import Image
from sklearn.metrics import confusion_matrix

class RootMaskTestDataset(Dataset):
    def __init__(self, mask_dir, label_dir, transform=None):
        self.mask_dir = mask_dir
        self.transform = transform
        self.samples = []

        # Build mapping from binary_mask name to crop label
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        mask_filenames = set(os.listdir(mask_dir))

        for label_file in label_files:
            with open(os.path.join(label_dir, label_file), 'r') as f:
                entries = json.load(f)

            for entry in entries:
                if entry["has_root"] != 1:
                    continue

                mask_name = entry["binary_mask"]
                matching_file = next((f for f in mask_filenames if f.endswith(mask_name)), None)

                if matching_file is not None:
                    self.samples.append((os.path.join(mask_dir, matching_file), entry["crop"]))

        # Get label to index mapping
        self.label_to_idx = {'Switchgrass': 0,
                             'Cotton': 1,
                             'Peanut': 2,
                             'Sesame': 3,
                             'Sunflower': 4,
                             'Papaya': 5}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mask_path, crop_label = self.samples[idx]
        mask_image = Image.open(mask_path).convert("L")  # Grayscale

        if self.transform:
            mask_image = self.transform(mask_image)

        label_idx = self.label_to_idx[crop_label]
        return mask_image, label_idx
    
class RootMaskTrainDataset(Dataset):
    def __init__(self, mask_dir, label_dir, transform=None):
        self.mask_dir = mask_dir
        self.transform = transform
        self.samples = []

        # Build mapping from binary_mask name to crop label
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        mask_filenames = set(os.listdir(mask_dir))

        for label_file in label_files:
            with open(os.path.join(label_dir, label_file), 'r') as f:
                entries = json.load(f)

            for entry in entries:
                if entry["has_root"] != 1:
                    continue

                mask_name = entry["binary_mask"]
                matching_file = next((f for f in mask_filenames if f.endswith(mask_name)), None)

                if matching_file is not None:
                    self.samples.append((os.path.join(mask_dir, matching_file), entry["crop"]))

        # Get label to index mapping
        self.label_to_idx = {'Switchgrass': 0,
                             'Cotton': 1,
                             'Peanut': 2,
                             'Sesame': 3,
                             'Sunflower': 4,
                             'Papaya': 5}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mask_path, crop_label = self.samples[idx]
        mask_image = Image.open(mask_path).convert("L")  # Grayscale

        if self.transform:
            mask_image = self.transform(mask_image)

        label_idx = self.label_to_idx[crop_label]
        return mask_image, label_idx

load_dotenv()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),       
    transforms.Normalize((0.5,), (0.5,)) 
])

scaler = GradScaler("cuda")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

experiment = Experiment(
    api_key=os.environ["COMET_API_KEY"],
    project_name="morphological",
    workspace="joannekim",
    auto_param_logging=False,
    auto_metric_logging=False,
    auto_output_logging=False
)

# Data
print('==> Preparing data..')

trainset = RootMaskTrainDataset('data/PRMI/masks/train/has_root', 'data/PRMI/labels/train', transform=transform)

testset = RootMaskTestDataset('data/PRMI/masks/test', 'data/PRMI/labels/test', transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=8, shuffle=True, num_workers=2)

label_counts = Counter()
for _, label in trainset:
    label_counts[label] += 1

print(label_counts)

num_classes = len(trainset.label_to_idx)
total_samples = len(trainset)
class_weights = {label: total_samples / (num_classes * count) for label, count in label_counts.items()}

sample_weights = [class_weights[label] for _, label in trainset]
sample_weights_tensor = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)

# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=8, sampler=sampler, num_workers=2)

trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=8, shuffle=True, num_workers=2)

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
# net = mnn_resnet.MNNResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

print(torch.cuda.device_count())

alpha_list = [class_weights[i] for i in range(6)] 
print(alpha_list)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    all_preds = []
    all_labels = []
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with autocast(device_type="cuda", dtype=torch.float16):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            if torch.isnan(outputs).any():
                print("NaN in outputs at epoch", epoch, "batch", batch_idx)
                print("Output stats:", outputs.min().item(), outputs.max().item())
                raise ValueError("NaN in model outputs")
            # loss = criterion(outputs, targets)
            alpha_tensor = torch.tensor(alpha_list).to(outputs.device)
            a_t = alpha_tensor[targets]

            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
            pt = torch.exp(-ce_loss)
            focal_loss = (a_t * (1-pt)**2 * ce_loss).mean()
            loss = focal_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(targets.cpu().tolist())

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    experiment.log_confusion_matrix(
        matrix=cm,
        labels=list(trainset.label_to_idx.keys()),
        title="Train Confusion Matrix",
        epoch=epoch,
        file_name='train.json'
    )
    experiment.log_metric('Loss', train_loss, epoch=epoch)

def test(epoch):
    global best_acc
    net.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # correct, total = 0, 0
    # heatmap = np.zeros((10, 2))      # rows: true CIFAR class, cols: pred 0/1

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs  = inputs.to(device)
            targets = targets.to(device)              # original 0-4 labels
            # targets_bin = (targets == 0).long()  # 0/1 on CUDA

            outputs = net(inputs)                     # shape [B, 2]
            _, predicted = outputs.max(1)             # 0/1

            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # update 5×2 heat-map
            # pred_np = predicted.cpu().numpy()
            # for t, p in zip(targets.cpu().numpy(), pred_np):
            #     heatmap[t, p] += 1

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())

            progress_bar(
                batch_idx, len(testloader),
                f'Acc: {100.*correct/total:.3f}% ({correct}/{total})'
            )

    # plot_heatmap(heatmap, experiment, epoch)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    experiment.log_confusion_matrix(
        matrix=cm,
        labels=list(testset.label_to_idx.keys()),
        title="Test Confusion Matrix",
        epoch=epoch,
        file_name='test.json'
    )

    acc = 100.0 * correct / total
    experiment.log_metric('Accuracy', acc, epoch=epoch)

    if acc > best_acc:
        print('Saving..')
        torch.save(
            {'net': net.state_dict(), 'acc': acc, 'epoch': epoch},
            './checkpoint/ckpt.pth'
        )
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    experiment.log_metrics(logger.log_weights(net), epoch=epoch)
    test(epoch)
    scheduler.step()
