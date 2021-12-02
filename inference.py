import os
import cv2
import numpy as np
import sys
import getopt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt


datapath = ''
out_csv_path = ''

argvs = sys.argv[1:]
try:
    opts, args = getopt.getopt(argvs, "", ["data=", "output="])
    for opt, arg in opts:
        if opt == '--data':
            datapath = arg
        elif opt == '--output':
            out_csv_path = arg
except getopt.GetoptError:
    print('inferenenc.py --data [data_path] --output [output_path]')
    sys.exit(2)
print(f'datapath {datapath} out_csv_path {out_csv_path}')

# url = 'https://drive.google.com/uc?id=1BzedlECiMt4n0Uc_s-jjbMegzpwsEOGq'
# output = 'model.zip'
# gdown.download(url, output, quiet=False)
# with zipfile.ZipFile('train.zip', 'r') as zip_ref:
#     zip_ref.extractall('')


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(0)

# Preprocessing code


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
    img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
    return img_crop


def roı_clahe_pre_process(folder, new_folder):
    for filename in os.listdir(folder):
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

        cropped = rotate_crop(img, min_area_rect)
        if(cropped.size == 0):
            print(f'cropped size 0 {filename}')
            continue

        # determine clahe values
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        cropped_clahe = clahe.apply(cropped)  # apply clahe transform on image

        # determine new path for save to image
        new_path = os.path.join(new_folder, filename)

        cv2.imwrite(new_path, cropped_clahe)  # save output to new paths


if(not os.path.isdir('./preprocessed')):
    os.mkdir('./preprocessed')
roı_clahe_pre_process('./test', './preprocessed')

test_tfm = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))  # 這邊還是用原始ImageNet的參數
])

# 設定root path
# train_root_path = os.path.join(datapath, 'train')  # '/content/train/'
test_root_path = os.path.join(datapath, 'test')  # '/content/test/'
# y_label_path = os.path.join(datapath, 'train.csv')
model_root_path = os.path.join(datapath, 'model')
# pandas_y = pd.read_csv(y_label_path, index_col='id').fillna(-1)

# # 將影像中的檔名，包含有不同部位的資料做成one hot encoding當作model判斷的輔助資料


def add_part_one_hot_encoding(y):
    hand_list = []
    forearm_list = []
    wrist_list = []

    for i in y.index.tolist():
        part = i.split('_')[0]
        if part == 'HAND':
            hand_list.append(1)
            forearm_list.append(0)
            wrist_list.append(0)

        elif part == 'FOREARM':
            hand_list.append(0)
            forearm_list.append(1)
            wrist_list.append(0)

        elif part == 'WRIST':
            hand_list.append(0)
            forearm_list.append(0)
            wrist_list.append(1)

        else:
            print(f'another part:{i}')

    column_names = ['label', 'HAND', 'FOREARM', 'WRIST']

    y = y.reindex(columns=column_names)
    y[column_names[0]] = y['label']
    y[column_names[1]] = hand_list
    y[column_names[2]] = forearm_list
    y[column_names[3]] = wrist_list

    return y


test_id = os.listdir(test_root_path)
test_label = [-1 for i in range(0, len(test_id))]
test_unlabel_y = pd.DataFrame(
    list(zip(test_id, test_label)), columns=['id', 'label'])
test_unlabel_y.set_index('id', inplace=True)
test_unlabel_y = add_part_one_hot_encoding(test_unlabel_y.copy())

patient_id_list = []
for i in test_unlabel_y.index:
    patient_id_list.append(i.split('_')[0]+'_'+i.split('_')[1])
test_unlabel_y = test_unlabel_y.assign(patient_id=patient_id_list)

# 建立customDataset的格式，主要是在pandas列表上做操作


class customDataset(Dataset):
    def __init__(self, pd_y_label, root_dir, mode, transform=None):
        self.pd_y_label = pd_y_label
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.pd_y_label.index[index]
        img_path = os.path.join(self.root_dir, self.mode, img_name)
        img = Image.open(img_path)
        y_label = self.pd_y_label.loc[img_name].tolist()[0]
        aux_x = self.pd_y_label.loc[img_name].tolist()[1:-1]

        if self.transform:
            img = self.transform(img)
        return img, aux_x, y_label

    def __len__(self):
        return len(self.pd_y_label)


batch_size = 64

test_set = customDataset(test_unlabel_y, test_root_path,
                         mode='test', transform=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 建立自己的model，主要更換後面的fc layer


class Classifier(nn.Module):
    def __init__(self, torch_pretrained_model):
        super(Classifier, self).__init__()

        self.torch_pretrained_model = torch_pretrained_model
        self.torch_pretrained_model.dropout = nn.Identity()
        self.torch_pretrained_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048+3),
            nn.Dropout(p=0.25, inplace=True),
            nn.Linear(in_features=2048+3, out_features=512, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(512, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, aux_x):
        x = self.torch_pretrained_model(x)
        # print(aux_x.shape,x.shape)
        x = torch.cat((aux_x, x), 1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


# ----------------------------------------- Inference(Testing) 用最好的model -------------------------------------------------------
torch_pretrained_model = torchvision.models.inception_v3(
    pretrained=False, init_weights=True, aux_logits=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = os.path.join(model_root_path+f"best_model_hand{12}.pt")

model = Classifier(torch_pretrained_model).to(device)

model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
model.device = device

model.eval()

predictions = []
for batch in tqdm(test_loader):
    imgs, aux_x, labels = batch
    aux_x = torch.cat((torch.reshape(aux_x[0], (len(aux_x[0]), 1)), torch.reshape(
        aux_x[1], (len(aux_x[1]), 1)), torch.reshape(aux_x[2], (len(aux_x[2]), 1))), 1).float()

    with torch.no_grad():
        logits = model(imgs.to(device), aux_x.to(device))
    predictions = predictions + logits.cpu().detach().numpy().reshape(-1).tolist()

with open("output.csv") as f:
    f.write("id,label\n")
    test_index = test_unlabel_y.index.tolist()
    for i, pred in enumerate(predictions):
        f.write(f"{test_index[i]},{pred}\n")
