# ========== [0] 패키지 import ==========
import os, glob, random, time, sys
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# 저장 디렉토리 생성
os.makedirs('./weights', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# ========== [1] 비디오 및 라벨 로딩 ==========
video_files = glob.glob('/home/elicer/ffpp_data/manipulated_sequences/Deepfakes/c23/videos/*.mp4') + \
              glob.glob('/home/elicer/ffpp_data/original_sequences/youtube/c23/videos/*.mp4')
random.shuffle(video_files)
labels = pd.read_csv('/home/elicer/DeepFake_Convolutional_LSTM/Gobal_metadata.csv', names=["file", "label"])

# ========== [2] Dataset 정의 ==========
class video_dataset(Dataset):
    def __init__(self, video_names, labels, sequence_length=60, transform=None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        filename = os.path.basename(video_path)
        label = self.labels.loc[self.labels["file"] == filename, "label"].values[0]
        label = 1 if label == 'FAKE' else 0

        for i, frame in enumerate(self.frame_extract(video_path)):
            if self.transform:
                frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)[:self.count]
        return frames, label

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success, image = vidObj.read()
        while success:
            yield image
            success, image = vidObj.read()

# ========== [3] 모델 정의 ==========
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, hidden_dim=2048):
        super(Model, self).__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.dp = nn.Dropout(0.4)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap).view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        out = self.dp(self.linear(torch.mean(lstm_out, dim=1)))
        return fmap, out

# ========== [4] 학습/검증 관련 함수 ==========
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1): self.val, self.sum, self.count = val, self.sum + val * n, self.count + n; self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    _, preds = outputs.max(1)
    return 100. * (preds == targets).sum().item() / targets.size(0)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        _, out = model(x)
        loss = criterion(out, y)
        acc = calculate_accuracy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))
    return loss_meter.avg, acc_meter.avg

def validate(model, loader, criterion):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            _, out = model(x)
            loss = criterion(out, y)
            acc = calculate_accuracy(out, y)
            _, preds = out.max(1)
            y_true += y.cpu().tolist()
            y_pred += preds.cpu().tolist()
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return y_true, y_pred, loss_meter.avg, acc_meter.avg, precision, recall

# ========== [5] 데이터 준비 ==========
im_size = 112
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
train_videos, val_videos = train_test_split(video_files, test_size=0.2)
train_data = video_dataset(train_videos, labels, transform=transform)
val_data = video_dataset(val_videos, labels, transform=transform)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4)

# ========== [6] 학습 실행 ==========
model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss().cuda()

train_losses, train_accs, val_losses, val_accs = [], [], [], []
precisions, recalls = [], []
epochs = 50
best_val_acc = 0.0

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")
    t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
    y_true, y_pred, v_loss, v_acc, v_prec, v_rec = validate(model, val_loader, criterion)

    train_losses.append(t_loss); train_accs.append(t_acc)
    val_losses.append(v_loss); val_accs.append(v_acc)
    precisions.append(v_prec); recalls.append(v_rec)

    if epoch == 1:
        torch.save(model.state_dict(), './weights/first_epoch_model.pth')
        print("📌 First epoch model saved")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), './weights/best_model.pth')
        print(f"✅ Best model saved at epoch {epoch} (val acc: {v_acc:.2f}%)")

torch.save(model.state_dict(), './weights/last_epoch_model.pth')
print("📌 Last epoch model saved")

# ========== [7] 시각화 저장 ==========
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend(); plt.title("Loss")
plt.savefig("./plots/loss_curve.png")

plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Val Accuracy")
plt.legend(); plt.title("Accuracy")
plt.savefig("./plots/accuracy_curve.png")

plt.figure()
plt.plot(precisions, label="Val Precision")
plt.plot(recalls, label="Val Recall")
plt.legend(); plt.title("Precision & Recall")
plt.savefig("./plots/precision_recall_curve.png")

cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=["FAKE", "REAL"], columns=["FAKE", "REAL"])
plt.figure()
sn.heatmap(df_cm, annot=True, fmt='g')
plt.title("Confusion Matrix")
plt.savefig("./plots/confusion_matrix.png")
