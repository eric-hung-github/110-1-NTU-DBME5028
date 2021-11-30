import sys
import getopt
import pandas as pd
import numpy as np
import os
import PIL
import cv2
from pandas.core.dtypes.missing import isna
from torchvision import models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

from tqdm import tqdm
from shutil import copyfile
from math import isnan

datapath = ''

argvs = sys.argv[1:]
try:
    opts, args = getopt.getopt(argvs, "", ["data="])
    for opt, arg in opts:
        if opt == '--data':
            datapath = arg
except getopt.GetoptError:
    print('train.py --data <data_path> ')
    sys.exit(2)
print(f'datapath{datapath}')


# Get cpu or gpu device for training.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# preprocess


def rotate_crop(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]
    return img_crop


def pre_process(folder, new_folder):
    if(not os.path.exists(new_folder)):
        os.mkdir(new_folder)
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename),
                         cv2.IMREAD_GRAYSCALE)  # read image from directory

        lower_black = np.array([0], dtype="uint16")
        upper_black = np.array([200], dtype="uint16")
        black_mask = cv2.inRange(img, lower_black, upper_black)
        img[np.where(black_mask == [0])] = [0]

        thresh = cv2.threshold(
            img, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(thresh, 1, 1)
        contours.sort(key=cv2.contourArea, reverse=True)

        min_area_rect = cv2.minAreaRect(contours[0][:, 0, :])

        cropped_img = rotate_crop(img, min_area_rect)
        if(cropped_img.size == 0):
            print(filename)
            continue

        # determine clahe values
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # apply clahe transform on image
        cropped_clahe = clahe.apply(cropped_img)

        # determine new path for save to image
        new_path = os.path.join(new_folder, filename)
        cv2.imwrite(new_path, cropped_clahe)  # save output to new paths
        # cv2.waitKey(delay=1000)


train_preprocced_folder = os.path.join(datapath, 'train/preprocced')
test_preprocced_folder = os.path.join(datapath, 'test/preprocced')
# run funnction with folder paths
pre_process(os.path.join(datapath, 'train'), train_preprocced_folder)
pre_process(os.path.join(datapath, 'test'), test_preprocced_folder)


train_df = pd.read_csv('train.csv')
train_dict = train_df.set_index('id').to_dict()['label']

images_types = ['labeled', 'unlabeled', 'new_labeled']

catalogies = ['HAND', 'WRIST', 'FOREARM']
catalogies_folder = {}
for catalogy in catalogies:
    catalogies_folder[catalogy] = os.path.join(
        train_preprocced_folder, 'folder_'+catalogy)


def cataloging(folder, catalogies):
    for catalog in catalogies:
        new_folder = os.path.join(folder, 'folder_'+catalog)
    if(not os.path.exists(new_folder)):
        os.mkdir(new_folder)

    images_list = os.listdir(folder)
    for catalog in catalogies:
        hand_imgs = [img_src for img_src in images_list if(
            img_src.startswith(catalog))]
        for hand_img in tqdm(hand_imgs):
            copyfile(os.path.join(folder, hand_img),
                     os.path.join(new_folder, hand_img))


cataloging(train_preprocced_folder, catalogies)


def seperate(images_folder, dict):
    # build dirctory for 3 images_typedirctory
    for images_type in images_types:
        if(not os.path.exists(os.path.join(images_folder, images_type))):
            os.mkdir(os.path.join(images_folder, images_type))
    images_list = os.listdir(images_folder)
    # catalogo image into labeled/unlabeled
    for id, label in tqdm(dict.items()):
        img_srcs = [
            img_src for img_src in images_list if img_src.startswith(id)]

        for img_src in img_srcs:
            file_src = os.path.join(images_folder, img_src)
            if os.path.isfile(file_src):
                os.rename(file_src, os.path.join(
                    images_folder, images_types[pd.isna(label)], img_src))


for catalog in catalogies:
    seperate(catalogies_folder[catalog], train_dict)

# transformers
proper_img_size = (256, 256)

transformers = {'train_transforms': transforms.Compose([
    transforms.Resize(proper_img_size),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
]),
    'test_transforms': transforms.Compose([
        transforms.Resize(proper_img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ]),
    'valid_transforms': transforms.Compose([
        transforms.Resize(proper_img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])}


class MyDataSet(Dataset):
    def __init__(self, images_folder_path, label_dic, transform=None):
        self.images_folder_path = images_folder_path
        self.images_list = os.listdir(images_folder_path)
        self.transform = transform
        self.label_dic = label_dic

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        id = '_'.join(image_name.split('_', 2)[:2])
        image = PIL.Image.open(os.path.join(
            self.images_folder_path, image_name)).convert('L')
        label = self.label_dic[id]
        if self.transform is not None:
            image = self.transform(image)
        if(isnan(label)):
            return image, id
        else:
            return image, int(label)


labeled_set = MyDataSet(os.path.join(
    catalogies_folder['HAND'], images_types[0]), train_dict, transformers['train_transforms'])

unlabeled_set = MyDataSet(os.path.join(
    catalogies_folder['HAND'], images_types[1]), train_dict, transformers['train_transforms'])

train_proportion = 0.9

# Compute size of Train/Validation
train_labeled_set_size = int(len(labeled_set) * train_proportion)
valid_labeled_set_size = len(labeled_set) - train_labeled_set_size

# Divide Dataset into Train/Validation
train_labeled_set, valid_labeled_set = torch.utils.data.random_split(
    labeled_set, [train_labeled_set_size, valid_labeled_set_size])

# Create data loaders.
batch_size = 32
num_workers = 2
pin_memory = True

train_labeled_dataloader = DataLoader(
    train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
valid_labeled_dataloader = DataLoader(
    valid_labeled_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

unlabeled_dataloader = DataLoader(
    unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


def train(dataloader, loss_function, optimizer):
    train_loss = 0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*len(labels)

    return train_loss/len(dataloader.dataset)


def valid(dataloader, loss_function):
    valid_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = nn.functional.softmax(model(inputs), dim=1)
            loss = loss_function(outputs, labels)
            valid_loss += loss.item()*len(labels)

            for i in range(len(labels)):
                max = 1 if outputs[i][1] > outputs[i][0] else 0
                accuracy += 1 if max == labels[i] else 0
    return (valid_loss/len(valid_labeled_dataloader.dataset), accuracy/len(valid_labeled_dataloader.dataset))


model = EfficientNet.from_pretrained(
    'efficientnet-b3', num_classes=2, in_channels=1)

MODEL_STATE_DIR = 'model/'
if(not os.path.exists(MODEL_STATE_DIR)):
    os.mkdir(MODEL_STATE_DIR)

IS_LOAD = 0
EPOCH_CHOOSEN = 19
LOAD_MODEL_SRC = os.path.join(MODEL_STATE_DIR, f'epoch_{EPOCH_CHOOSEN}.pth')

if(IS_LOAD):
    model.load_state_dict(torch.load(LOAD_MODEL_SRC))

model.to(device)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        if(weight is not None):
            # weight parameter will act as the alpha parameter to balance class weights
            self.weight = weight.to(device)

    def forward(self, input, target):

        ce_loss = nn.functional.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


loss_function = FocalLoss(weight=torch.tensor([1.0, 2.0]))


optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-2)

# The number of training epochs.
n_epochs = 20

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, eta_min=0, T_max=n_epochs)

for epoch in range(n_epochs):
    train_loss = train(train_labeled_dataloader, loss_function, optimizer)
    valid_loss, accuracy = valid(valid_labeled_dataloader, loss_function)
    print(f'[{epoch} epoch] Train_loss:{train_loss} Valid_loss:{valid_loss} Accuracy:{accuracy}')
    torch.save(model.state_dict(), os.path.join(
        MODEL_STATE_DIR, f'epoch_{epoch}.pth'))

PROPERBILITY = 0.95


def labeling(dataloader, properbility):
    new_labeling_dic = {}
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)

            for i in range(len(outputs)):
                if(outputs[i][0].item() > properbility):
                    new_labeling_dic[labels[i]] = 0
                elif(outputs[i][1].item() > properbility):
                    new_labeling_dic[labels[i]] = 1
    return new_labeling_dic


# copy newlabeling to directory
new_labeling_dict = labeling(unlabeled_dataloader, PROPERBILITY)
new_labeling_dir = os.path.join(catalogies_folder['HAND'], images_types[2])
newlabeled_images_list = os.listdir(new_labeling_dir)

for id, label in tqdm(new_labeling_dict.items()):
    img_srcs = [
        img_src for img_src in newlabeled_images_list if img_src.startswith(id)]

    for img_src in img_srcs:
        org_src = os.path.join(
            catalogies_folder['HAND'], images_types[1], img_src)
        if os.path.isfile(org_src):
            copyfile(org_src, os.path.join(new_labeling_dir, img_src))

new_labeled_set = MyDataSet(
    new_labeling_dir, new_labeling_dict, transformers['train_transforms'])

new_labeled_dataloader = DataLoader(
    new_labeled_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

for epoch in range(n_epochs):
    train_loss = train(new_labeled_dataloader, loss_function, optimizer)
    test_loss, accuracy = valid(valid_labeled_dataloader, loss_function)
    print(f'[{epoch} epoch] Train_loss:{train_loss/len(train_labeled_dataloader.dataset)} Test_loss:{test_loss/len(valid_labeled_dataloader.dataset)} Accuracy:{accuracy/len(valid_labeled_dataloader.dataset)}')
    torch.save(model.state_dict(), os.path.join(
        MODEL_STATE_DIR, f'epoch_{epoch}.pth'))


def submit(dataloader, data_dic):
    result_dict = {}
    for key in data_dic:
        result_dict[key] = (0, 0)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = nn.functional.softmax(model(inputs), dim=1)

            for i in range(len(labels)):
                result_dict[labels[i]][j] += outputs[i][j]
    result_list = []
    for key, value in result_dict.items():
        nn.functional.softmax(value, dim=1)
        result_list.append(key, value[1])
    return result_list


TEST_DIR = 'test/preprocced'

test_dict = {}
newlabeled_images_list = os.listdir(TEST_DIR)
for image_name in newlabeled_images_list:
    id = '_'.join(image_name.split('_', 2)[:2])
    test_dict[id] = np.nan

test_set = MyDataSet(TEST_DIR, test_dict, transformers['test_transforms'])
test_dataloader = DataLoader(test_set, batch_size=batch_size)

anser_list = submit(test_dataloader, test_dict)
anser_df = pd.DataFrame(data=anser_list, columns=['id', 'label'])
anser_df.to_csv('submit.csv', index=False)
