from data import read_img_tsv, read_navigable_tsv
from models import LinearBinaryModel
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import time

print("Loading Image Features")
img_data = read_img_tsv("img_features/ResNet-152-imagenet.tsv")
print("Loading Nav Features")
nav_data = read_navigable_tsv("img_features/navigable.tsv")

DS_SIZE = len(nav_data)

train_len = int(DS_SIZE * 0.8)

train_ds = zip(img_data[:train_len], nav_data[:train_len])
val_ds = zip(img_data[train_len:], nav_data[train_len:])

def arrayify(z): return np.array(list(z))

train_ds = arrayify(train_ds)
val_ds = arrayify(val_ds)

max_epoch = 200

thresholding = 0.2

# model = LinearBinaryModel(2048)
# model = LinearBinaryModel(2048, 256)
model = LinearBinaryModel(2048, 256, 128).cuda()
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
writer = SummaryWriter('tensorboard_logs/pointing-2048-biggestboi-c-0.2')

for i in range(max_epoch):
    # train
    # a = time.clock()
    epoch_train_ds = np.random.permutation(train_ds)
    # print(time.clock() - a, "permute")
    all_losses = []
    a = time.clock()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for img, nav in epoch_train_ds:
        x = torch.tensor(img['features']).cuda()
        y = torch.tensor(nav['nav']).cuda()
        preds = model(x)

        loss = criterion(preds, y)
        all_losses.append(loss.detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_hard = y > 0.5
        preds_hard = preds > thresholding
        TP += torch.sum(y_hard & preds_hard).item()
        TN += torch.sum(~y_hard & ~preds_hard).item()
        FP += torch.sum(~y_hard & preds_hard).item()
        FN += torch.sum(y_hard & ~preds_hard).item()

    total = TP + TN + FP + FN

    print(TP/total)
    print(TN/total)
    print(FP/total)
    print(FN/total)

    print(time.clock() - a, "comp")
    avg_loss = np.average(all_losses)
    writer.add_scalar('Loss/train', avg_loss, i)
    writer.add_scalar('Accuracy/train', (TP + TN) / total, i)
    writer.add_scalar('Passes/train', (TP + FP) / total, i)
    writer.add_scalar('Precision/train', TP / (TP + FP), i)
    writer.add_scalar('Recall/train', TP / (TP + FN), i)

    # val
    if (i + 1) % 10 == 0:
        print("Epoch " + str(i))
        print("Train Loss: " + str(avg_loss))
        epoch_val_ds = np.random.permutation(val_ds)
        print("evaling")
        all_losses = []
        
        for img, nav in epoch_val_ds:
            x = torch.tensor(img['features']).cuda()
            y = torch.tensor(nav['nav']).cuda()
            preds = model(x)

            loss = criterion(preds, y)
            all_losses.append(loss.item())

            y_hard = y > 0.5
            preds_hard = preds > thresholding
            TP += torch.sum(y_hard & preds_hard).item()
            TN += torch.sum(~y_hard & ~preds_hard).item()
            FP += torch.sum(~y_hard & preds_hard).item()
            FN += torch.sum(y_hard & ~preds_hard).item()

        total = TP + TN + FP + FN

        print(TP/total)
        print(TN/total)
        print(FP/total)
        print(FN/total)

        print(time.clock() - a, "comp")
        avg_loss = np.average(all_losses)
        writer.add_scalar('Loss/val', avg_loss, i)
        writer.add_scalar('Accuracy/val', (TP + TN) / total, i)
        writer.add_scalar('Passes/val', (TP + FP) / total, i)
        writer.add_scalar('Precision/val', TP / (TP + FP), i)
        writer.add_scalar('Recall/val', TP / (TP + FN), i)
