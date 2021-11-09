# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16 --epochs 2 --trials 2 --iterations 2 --dataset_dir ../Datasets
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

import pdb, os, glob, sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger, make_prediction
from util.cutout import Cutout
import util.file_utils as file_utils

from model.resnet import ResNet18
from model.wide_resnet import WideResNet

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--dataset_dir', default='Data', type=str,
                    help='The location of the dataset to be explored')
parser.add_argument('--trials', default=5, type=int,
                    help='Number of times to run the complete experiment')
parser.add_argument('--iterations', default=2, type=int,
                    help='Number of times to run the complete experiment')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

test_id = args.dataset + '_' + args.model

print(args)

if not os.path.exists(args.dataset_dir):
    file_utils.create_dir(args.dataset_dir)

dataset_list = sorted(glob.glob(args.dataset_dir + "/*"))
print("Dataset List: ", dataset_list)

if len(dataset_list) == 0:
    print("ERROR: 1. Add the Datasets to be run inside of the", args.dataset_dir, "folder")
    sys.exit()

if args.data_augmentation:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


for dataset in dataset_list:

    # 1. Location to save the output for the given dataset
    current_dataset_file = dataset.split("/")[-1] + '_.txt'

    for iteration in range(args.iterations):

        #2. Prepare the training and test data
        train_dataset = datasets.ImageFolder(os.path.join(dataset, 'train'),
                                              train_transform)

        test_dataset = datasets.ImageFolder(os.path.join(dataset, 'test'),
                                              test_transform)
        num_classes = len(train_dataset.classes)


        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=2)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

        # Iterate over the trials
        for trial in range(args.trials):

            print("Test result for iteration", iteration, " experiment:", trial, "for dataset", dataset)

            # Create new model for each trial
            cnn = ResNet18(num_classes=num_classes)

            cnn = cnn.cuda()
            criterion = nn.CrossEntropyLoss().cuda()
            cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                            momentum=0.9, nesterov=True, weight_decay=5e-4)

            scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

            log_dir = 'logs/'
            checkpoint_dir = 'checkpoints/'

            file_utils.create_dir(log_dir)
            file_utils.create_dir(checkpoint_dir)

            filename = log_dir + test_id + '.csv'
            csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

            for epoch in range(args.epochs):

                xentropy_loss_avg = 0.
                correct = 0.
                total = 0.

                progress_bar = tqdm(train_loader)
                for i, (images, labels) in enumerate(progress_bar):
                    progress_bar.set_description('Epoch ' + str(epoch))

                    images = images.cuda()
                    labels = labels.cuda()

                    cnn.zero_grad()
                    pred = cnn(images)

                    xentropy_loss = criterion(pred, labels)
                    xentropy_loss.backward()
                    cnn_optimizer.step()

                    xentropy_loss_avg += xentropy_loss.item()

                    # Calculate running average of accuracy
                    pred = torch.max(pred.data, 1)[1]
                    total += labels.size(0)
                    correct += (pred == labels.data).sum().item()
                    accuracy = correct / total

                    progress_bar.set_postfix(
                        xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                        acc='%.3f' % accuracy)

                test_acc = test(test_loader)
                tqdm.write('test_acc: %.3f' % (test_acc))

                # scheduler.step(epoch)  # Use this line for PyTorch <1.4
                scheduler.step()     # Use this line for PyTorch >=1.4

                row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
                csv_logger.writerow(row)

                if epoch + 1 == args.epochs:
                    with open(current_dataset_file, 'a') as f:
                        print("Test result for iteration", iteration, " experiment:", trial, "for dataset", dataset, file = f)
                        print(make_prediction(cnn, test_dataset.classes, test_loader, 'save'), file = f)

            torch.save(cnn.state_dict(), checkpoint_dir + test_id + '.pt')
            csv_logger.close()
