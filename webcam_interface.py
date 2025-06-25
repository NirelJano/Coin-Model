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

# הוספת מילון לערכי המטבעות
COIN_VALUES = {
    "One": 1,
    "Two": 2,
    "Five": 5,
    "Ten": 10
}

# טען את המודל עם num_classes נכון
model = RFDETRBase(
    num_classes=4,
    pretrain_weights="checkpoint_best_ema.pth"
)

# אתחול העוקב
tracker = sv.ByteTrack()

# אתחול משתנים לסכימה
counted_coins = {} # מילון לשמירת המטבעות שנספרו, עם tracker_id כמפתח
total_sum = 0

cap = cv2.VideoCapture(0)
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

    # עדכון העוקב עם הזיהויים החדשים
    detections = tracker.update_with_detections(detections)

    # חישוב הסכום ומניעת ספירה כפולה
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        # בדוק אם המטבע הזה (עם ה-tracker_id שלו) כבר נספר
        if tracker_id not in counted_coins:
            coin_name = CUSTOM_CLASSES[class_id]
            if coin_name in COIN_VALUES:
                coin_value = COIN_VALUES[coin_name]
                counted_coins[tracker_id] = coin_value
                # עדכן את הסכום הכולל
                total_sum = sum(counted_coins.values())

    # הכנת תוויות עם tracker_id
    labels = [
        f"#{tid} {CUSTOM_CLASSES[cid]} {conf:.2f}"
        for cid, conf, tid in zip(detections.class_id, detections.confidence, detections.tracker_id)
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

    # הצגת הסכום הכולל על המסך
    sum_text = f"Total Sum: {total_sum} ILS"
    cv2.putText(annotated, sum_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
