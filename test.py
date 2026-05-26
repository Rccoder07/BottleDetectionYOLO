from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(
    r"C:\Users\rudra\BottleDetection\runs\detect\train-5\weights\best.pt"
)

# Clear bottle image
image_path = r"C:\Users\rudra\OneDrive\Desktop\DESKTOP\STUDY\KJSCE\INTERNSHIP\INHOUSE\2_NINAD MEHENDALE SIR\DAY-6\BottleDetection\dataset\images\bottle\shopping.webp"

# Run detection
results = model.predict(source=image_path, conf=0.05)

# Draw detections
annotated_frame = results[0].plot()

# Show output
cv2.imshow("Bottle Detection", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()