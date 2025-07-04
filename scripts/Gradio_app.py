import gradio as gr
import torch
from model import ConvLSTMClassifier
from dataset_loader import get_transform
import cv2
import os
import subprocess
from torchvision import transforms
from PIL import Image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTMClassifier().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = get_transform()

def ensure_mp4(video_path):
    # If already mp4, return as is
    if video_path.lower().endswith('.mp4'):
        return video_path, False
    # Convert to mp4 using ffmpeg
    tmp_mp4 = video_path + "_converted.mp4"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-strict", "experimental",
        tmp_mp4
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return tmp_mp4, True
    except Exception as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e}")

def extract_frames_from_video(video_path, sequence_length=16, is_fight=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // sequence_length)
    frames = []
    count = 0
    saved = 0
    while cap.isOpened() and saved < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            # Annotate frame
            frame = annotate_frame(frame, is_fight)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            pil_img = transform(pil_img)
            frames.append(pil_img)
            saved += 1
        count += 1
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames extracted. The video may be corrupted or in an unsupported format.")
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    clip_tensor = torch.stack(frames)  # (seq, C, H, W)
    return clip_tensor.unsqueeze(0)  # (1, seq, C, H, W)

def predict_video(video_file):
    try:
        mp4_path, is_converted = ensure_mp4(video_file)
        clip = extract_frames_from_video(mp4_path).to(device)
        with torch.no_grad():
            outputs = model(clip)
            pred = torch.argmax(outputs, dim=1).item()
        result = "Fight Detected!" if pred == 1 else "No Fight Detected."
        # Return the mp4 path for preview, and the result
        return mp4_path, result
    except Exception as e:
        return None, f"Error: {str(e)}"

def annotate_frame(frame, is_fight):
    color = (0, 0, 255) if is_fight else (0, 255, 0)  # Red for fight, green for non-fight
    label = "Fight" if is_fight else "NonFight"
    thickness = 3

    # Draw rectangle (full frame or ROI as needed)
    h, w, _ = frame.shape
    cv2.rectangle(frame, (10, 10), (w-10, h-10), color, thickness)
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return frame

iface = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload a video"),
    outputs=[gr.Video(label="Playable Video"), gr.Textbox(label="Prediction")],
    title="Anomaly (Fight) Detection",
    description="Upload a video and the model will predict if a fight is detected."
)

if __name__ == "__main__":
    iface.launch()