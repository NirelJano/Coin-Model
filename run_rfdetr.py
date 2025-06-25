import argparse
import supervision as sv
import os
from PIL import Image
from rfdetr import RFDETRBase

# הגדרת המחלקות התואמות לדאטהסט שלך
# הסדר חשוב - הוא צריך להתאים לאיך שהמודל אומן
CUSTOM_CLASSES = [
  None,     # אינדקס 0
  "Five",   # אינדקס 1
  "One",    # אינדקס 2
  "Ten",    # אינדקס 3
  "Two",    # אינדקס 4
]
def main(image_path: str, weights_path: str, threshold: float):
    """
    Runs object detection on an image and displays the annotated result.

    Args:
        image_path (str): Path to the input image.
        weights_path (str): Path to the model weights file.
        threshold (float): Detection confidence threshold.
    """
    # 1. טען את המודל עם המשקולות
    print("Loading model...")
    model = RFDETRBase(pretrain_weights=weights_path,num_classes=4)

    # 2. טען תמונה
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # 3. בצע ניבוי
    print("Running prediction...")
    detections = model.predict(image, threshold=threshold)

    # 4. הגדר תוויות
    labels = [
        f"{CUSTOM_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # 5. אתר וסמן את התמונה
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

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels=labels)

    # 6. הצג את התמונה
    print("Displaying result.")
    sv.plot_image(annotated_image)

    # 7. שמור את התמונה
    base_name, ext = os.path.splitext(image_path)
    output_path = f"{base_name}_annotated{ext}"
    annotated_image.save(output_path)
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF-DETR object detection.")
    parser.add_argument(
        "--image-path",
        type=str,
        default="/Users/shachafemanoel/Documents/Coin-Model/coins test/IMG_9200.jpeg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="/Users/shachafemanoel/Documents/Coin-Model/checkpoint_best_ema.pth",
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold.",
    )
    args = parser.parse_args()
    
    main(args.image_path, args.weights_path, args.threshold)
