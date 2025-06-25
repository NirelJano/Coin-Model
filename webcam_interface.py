import cv2
import supervision as sv
from rfdetr import RFDETRBase

CUSTOM_CLASSES = [
  None,
  "Five",
  "One",
  "Ten",
  "Two",
]

# טען את המודל עם num_classes נכון
model = RFDETRBase(
    num_classes=4,
    pretrain_weights="checkpoint_best_ema.pth"
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("לא הצליח לפתוח את מצלמת הרשת")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break

    # תקן את ה-negative-stride ו-BGR→RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # או: rgb = frame[:, :, ::-1].copy()

    # הרץ ניבוי
    detections = model.predict(rgb, threshold=0.0) 
    min_conf = 0.5
    detections = detections[detections.confidence > min_conf]

    # הכנת תוויות
    labels = [
        f"{CUSTOM_CLASSES[cid]} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    # אנוטציה
    ts = sv.calculate_optimal_text_scale(resolution_wh=(frame.shape[1], frame.shape[0]))
    th = sv.calculate_optimal_line_thickness(resolution_wh=(frame.shape[1], frame.shape[0]))
    boxer = sv.BoxAnnotator(thickness=th)
    liner = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=ts,
        text_thickness=th,
        smart_position=True
    )

    annotated = boxer.annotate(frame, detections)
    annotated = liner.annotate(annotated, detections, labels)

    cv2.imshow("Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
