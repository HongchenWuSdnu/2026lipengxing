import cv2
import os

# ====== 路径设置（你只需要改这里） ======
video_path = "test.mp4"          # 视频路径
output_dir = "frames"            # 输出图片文件夹
frame_interval = 10              # 每隔多少帧保存一张
# =======================================

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        img_name = f"frame_{saved_id:04d}.jpg"
        img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(img_path, frame)
        saved_id += 1

    frame_id += 1

cap.release()
print(f"抽帧完成，共保存 {saved_id} 张图片")

