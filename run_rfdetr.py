from rfdetr import RFDETRBase
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

# === הגדרות נתיב ===
MODEL_PATH = "/Users/innadaymand/Desktop/coin model/results/output/checkpoint_best_ema.pth"
IMAGE_PATH = "/Users/innadaymand/Desktop/coin model/coins test/IMG_4547.jpeg"

# === שמות הקלאסים (עדכון לפי הסדר שלך!) ===
CLASS_NAMES = ["Five", "Ten", "One", 'Two', "0.5Shekel"]

# === טען את המודל ===
model = RFDETRBase(pretrain_weights=MODEL_PATH)

# === טען תמונה והריץ זיהוי ===
image = Image.open(IMAGE_PATH).convert("RGB")
detections = model.predict(image, threshold=0.5)

# === תוצאה לבדיקה ===
print("Detected class IDs:", detections.class_id)
print("Confidence scores:", detections.confidence)

# === ציור התוצאה ===
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = ImageFont.load_default()

for box, class_id, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
    x1, y1, x2, y2 = map(int, box)
    label_index = int(class_id)
    if label_index >= len(CLASS_NAMES):
        label = f"class_{label_index}"
    else:
        label = CLASS_NAMES[label_index]
    conf_percent = f"{conf * 100:.1f}%"
    text = f"{label} ({conf_percent})"

    # חישוב גודל טקסט בעזרת textbbox
    text_bbox = draw.textbbox((x1, y1), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # ציור מלבן וטקסט
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="black")
    draw.text((x1, y1 - text_height), text, fill="white", font=font)

# === שמור והצג ===
annotated.save("annotated_with_labels_and_conf.jpg")
annotated.show()
