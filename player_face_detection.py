from ultralytics import YOLO
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="video 1.mp4")
parser.add_argument("--output", default="results")
args = parser.parse_args()

if not os.path.exists(args.source):
    print("Error: video file not found")
    exit()

model = YOLO("yolov8n.pt")

model.predict(
    source=args.source,
    conf=0.4,
    save=True,
    project=args.output,
    name="predict"
)