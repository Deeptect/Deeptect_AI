import os
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import face_recognition
from tqdm import tqdm

# ----- 모델 정의 -----
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x, None)
        out = self.dp(self.linear(x_lstm[:, -1, :]))
        return fmap, out

# ----- 유틸리티 -----
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)

transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict(model, img, threshold=0.5):
    _, logits = model(img.cuda())
    logits = sm(logits)
    fake_prob = logits[:, 1].item()
    prediction = 1 if fake_prob >= threshold else 0
    return prediction, fake_prob

# ----- Dataset 정의 -----
class ValidationDataset(Dataset):
    def __init__(self, video_paths, sequence_length=20, transform=None, use_face_crop=True):
        self.video_paths = video_paths
        self.transform = transform
        self.count = sequence_length
        self.use_face_crop = use_face_crop

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(path)):
            if i % a == first_frame:
                if self.use_face_crop:
                    faces = face_recognition.face_locations(frame)
                    if faces:
                        top, right, bottom, left = faces[0]
                        frame = frame[top:bottom, left:right, :]
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        cap = cv2.VideoCapture(path)
        success = True
        while success:
            success, frame = cap.read()
            if success:
                yield frame

# ----- 실행 예시 -----
if __name__ == "__main__":
    threshold = 0.5
    model_path = "/home/elicer/DeepFake_Convolutional_LSTM/checkpoints2/best_model.pth"
    real_videos = sorted(os.listdir("/home/elicer/ffpp_data/original_sequences/youtube/c23/videos"))[:50]
    fake_videos = sorted(os.listdir("/home/elicer/ffpp_data/manipulated_sequences/Deepfakes/c23/videos"))[:50]

    real_paths = [f"/home/elicer/ffpp_data/original_sequences/youtube/c23/videos/{f}" for f in real_videos]
    fake_paths = [f"/home/elicer/ffpp_data/manipulated_sequences/Deepfakes/c23/videos/{f}" for f in fake_videos]

    all_paths = real_paths + fake_paths
    all_labels = [0]*50 + [1]*50  # 0: REAL, 1: FAKE

    dataset = ValidationDataset(all_paths, sequence_length=20, transform=transform_pipeline)
    model = Model(num_classes=2).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    for i in tqdm(range(len(all_paths))):
        pred, prob = predict(model, dataset[i], threshold=threshold)
        label = all_labels[i]
        print(f"{os.path.basename(all_paths[i])}: GT={label}, Pred={pred}, Prob={prob:.3f}")
        if pred == label:
            correct += 1

    acc = correct / len(all_paths) * 100
    print(f"\n✅ Accuracy: {acc:.2f}% ({correct}/{len(all_paths)})")
