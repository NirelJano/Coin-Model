import supervision as sv
import os
import sys
import numpy as np
import cv2
from PIL import Image
from rfdetr import RFDETRBase

# Add the script's directory to the Python path to ensure robust imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import CUSTOM_CLASSES, COIN_VALUES

# --- Configuration ---
# הגדר כאן את הנתיבים והפרמטרים הרצויים
WEIGHTS_PATH = "checkpoint_best_ema.pth"
IMAGE_PATH = "coins test/IMG_9200.jpeg"
CONFIDENCE_THRESHOLD = 0.4
# ---------------------

def main(image_path: str, weights_path: str, threshold: float):
    """
    Runs object detection on an image and displays the annotated result.

    Args:
        image_path (str): Path to the input image.
        weights_path (str): Path to the model weights file.
        threshold (float): Detection confidence threshold.
    """
    # 1. טען את המודל עם המשקולות
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return
    print("Loading model...")
    model = RFDETRBase(pretrain_weights=weights_path, num_classes=len(CUSTOM_CLASSES) - 1)

    # 2. טען תמונה
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # 3. בצע ניבוי
    print("Running prediction...")
    detections = model.predict(image, threshold=threshold)
    min_conf = 0.5
    detections = detections[detections.confidence > min_conf]

    # 4. חשב את הסכום הכולל של המטבעות שזוהו
    total_sum = sum(COIN_VALUES.get(CUSTOM_CLASSES[class_id], 0) for class_id in detections.class_id)
    sum_text = f"Total Sum: {total_sum} ILS"
    print(sum_text)

    # 5. הגדר תוויות לכל זיהוי
    labels = [
        f"{CUSTOM_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # 6. אתר וסמן את התמונה
    print("Annotating image...")
    # חישוב פרמטרים אופטימליים לאנוטציה
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )

    # המרת תמונת PIL למערך NumPy עבור אנוטציה
    annotated_frame = np.array(image.copy())
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)

    # הוספת טקסט הסכום הכולל במרכז התמונה
    font_scale_for_sum = text_scale * 1.5
    (text_width, text_height), _ = cv2.getTextSize(
        text=sum_text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale_for_sum,
        thickness=thickness
    )
    image_width, _ = image.size
    text_x = (image_width - text_width) // 2
    text_y = 60  # Padding from the top

    annotated_frame = sv.draw_text(
        scene=annotated_frame,
        text=sum_text,
        text_anchor=sv.Point(x=text_x, y=text_y),
        text_color=sv.Color.BLACK,
        text_scale=font_scale_for_sum,
        text_thickness=thickness,
        background_color=sv.Color.WHITE
    )

    # 7. הצג את התמונה
    print("Displaying result.")
    sv.plot_image(annotated_frame)

    # 8. שמור את התמונה
    final_image_pil = Image.fromarray(annotated_frame)
    base_name, ext = os.path.splitext(image_path)
    output_path = f"{base_name}_annotated{ext}"
    final_image_pil.save(output_path)
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    main(IMAGE_PATH, WEIGHTS_PATH, CONFIDENCE_THRESHOLD)
