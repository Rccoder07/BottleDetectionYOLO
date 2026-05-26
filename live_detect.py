from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(
    r"C:\C:\Users\rudra\OneDrive\Desktop\DESKTOP\STUDY\KJSCE\INTERNSHIP\INHOUSE\2_NINAD MEHENDALE SIR\DAY-6\BottleDetection\runs\detect\train-5\weights\best.pt"
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect bottles
    results = model.predict(source=frame, conf=0.05)

    # Draw boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Live Bottle Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()