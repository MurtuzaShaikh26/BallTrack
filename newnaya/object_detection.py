import cv2
import os
from ultralytics import YOLO

# Load your YOLO model
model = YOLO("best.pt")  # Replace with your trained model

# Video input-output pairs
videos = [
    ("broadcast.mp4", "outputs/broadcast_tracked.avi"),
    ("tacticam.mp4", "outputs/tacticam_tracked.avi")
]

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Tracking configuration
tracker_args = {
    'persist': True,
    'conf': 0.5,
    'iou': 0.5,
    'tracker': 'bytetrack.yaml',  # Uses the built-in ByteTrack configuration
}

for vid_in, vid_out in videos:
    cap = cv2.VideoCapture(vid_in)
    if not cap.isOpened():
        print(f"❌ Cannot open {vid_in}")
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_out, fourcc, fps, (width, height))

    frame_id = 0
    print(f"📹 Tracking on: {vid_in}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # Run tracking inference with Ultralytics
        results = list(model.track(
            source=frame,
            stream=True,
            persist=tracker_args['persist'],
            conf=tracker_args['conf'],
            iou=tracker_args['iou'],
            tracker=tracker_args['tracker']
        ))

        # Annotated frame with boxes and track IDs
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        if frame_id % 50 == 0:
            print(f"🧮 Frame {frame_id} processed")

    cap.release()
    out.release()
    print(f"✅ Saved tracked video to {vid_out}")
