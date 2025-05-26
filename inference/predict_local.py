import os
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import face_recognition

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
inv_normalize = transforms.Normalize(
    mean=-1 * np.divide(mean, std),
    std=np.divide([1, 1, 1], std)
)

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().squeeze()
    image = inv_normalize(image)
    image = image.numpy().transpose(1, 2, 0).clip(0, 1)
    return image

def predict(model, img, save_path="./output_heatmap3.png", threshold=0.7):
    fmap, logits = model(img.cuda())
    weight_softmax = model.linear.weight.detach().cpu().numpy()
    logits = sm(logits)
    fake_prob = logits[:, 1].item()
    prediction = 1 if fake_prob >= threshold else 0
    confidence = fake_prob * 100

    print(f"Fake prob: {fake_prob:.4f} → Prediction: {'FAKE' if prediction else 'REAL'} (Threshold: {threshold})")

    idx = prediction
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T, weight_softmax[idx, :].T)
    predict_map = np.uint8(255 * (out.reshape(h, w) - out.min()) / (out.max() - out.min()))
    heatmap = cv2.applyColorMap(cv2.resize(predict_map, (112, 112)), cv2.COLORMAP_JET)
    img_np = im_convert(img[:, -1, :, :, :])
    result = (heatmap * 0.5 + img_np * 0.8 * 255).astype(np.uint8)
    cv2.imwrite(save_path, result)

    return prediction, confidence

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
    im_size = 112
    threshold = 0.7
    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    path_to_videos = ["/home/elicer/ffpp_data/original_sequences/youtube/c23/videos/026.mp4"]
    model_path = "/home/elicer/DeepFake_Convolutional_LSTM/checkpoints2/best_model.pth"

    dataset = ValidationDataset(path_to_videos, sequence_length=20, transform=transform_pipeline)
    model = Model(num_classes=2).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i in range(len(path_to_videos)):
        print(f"Processing: {path_to_videos[i]}")
        prediction, conf = predict(model, dataset[i], save_path=f"./output_{i}.png", threshold=threshold)
        print("Result:", "REAL" if prediction == 0 else "FAKE")
