import cv2
import os


# ===== 参数区 =====
video_dir = "dataset_raw/videos"     # 放视频的目录
output_dir = "dataset_frames/test"  # train / val / test
label = "fake"                       # real or fake
frames_per_video = 10                # 每个视频抽多少帧
# ==================

save_dir = os.path.join(output_dir, label)
os.makedirs(save_dir, exist_ok=True)

video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

for vid in video_files:
    video_path = os.path.join(video_dir, vid)
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        continue

    step = max(frame_count // frames_per_video, 1)
    saved = 0
    fid = 0

    while saved < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break

        if fid % step == 0:
            name = f"{vid.replace('.mp4','')}_{saved:02d}.jpg"
            cv2.imwrite(os.path.join(save_dir, name), frame)
            saved += 1
        fid += 1

    cap.release()
    print(f"{vid} -> 保存 {saved} 帧")
