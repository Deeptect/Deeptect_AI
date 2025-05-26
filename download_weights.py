import os
import gdown

# 다운로드 경로 준비
os.makedirs("weights", exist_ok=True)

# 파일 ID를 붙여서 URL 구성
file_id = "1d0uZ9yQAW6Nawqxzj05Om4_vqj6k78kK"
url = f"https://drive.google.com/uc?id={file_id}"

output = "weights/best_model.pth"
gdown.download(url, output, quiet=False)