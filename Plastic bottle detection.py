import cv2
import os

# ==============================
# SETTINGS
# ==============================

IMAGE_DIR = "dataset/images/bottle"
LABEL_DIR = "dataset/annotations"

CLASS_ID = 0

os.makedirs(LABEL_DIR, exist_ok=True)

# ==============================
# GLOBAL VARIABLES
# ==============================

drawing = False
ix, iy = -1, -1
boxes = []

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

# ==============================
# MOUSE FUNCTION
# ==============================

def draw_rectangle(event, x, y, flags, param):

    global ix, iy, drawing, boxes, img_display

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            img_display = resized_img.copy()

            cv2.rectangle(
                img_display,
                (ix, iy),
                (x, y),
                (0, 255, 0),
                2
            )

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        cv2.rectangle(
            img_display,
            (ix, iy),
            (x, y),
            (0, 255, 0),
            2
        )

        boxes.append((ix, iy, x, y))

# ==============================
# MAIN LOOP
# ==============================

for image_name in os.listdir(IMAGE_DIR):

    image_path = os.path.join(IMAGE_DIR, image_name)

    img = cv2.imread(image_path)

    if img is None:

        print(f"Skipping {image_name}")
        continue

    original_h, original_w = img.shape[:2]

    # ==============================
    # FIX SAME WINDOW SIZE
    # ==============================

    scale = min(
        WINDOW_WIDTH / original_w,
        WINDOW_HEIGHT / original_h
    )

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized_img = cv2.resize(img, (new_w, new_h))

    img_display = resized_img.copy()

    boxes = []

    cv2.namedWindow("Annotator")

    cv2.setMouseCallback("Annotator", draw_rectangle)

    while True:

        cv2.imshow("Annotator", img_display)

        key = cv2.waitKey(1)

        # ==============================
        # SAVE
        # ==============================

        if key == ord('s'):

            label_path = os.path.join(
                LABEL_DIR,
                os.path.splitext(image_name)[0] + ".txt"
            )

            with open(label_path, "w") as f:

                for box in boxes:

                    x1, y1, x2, y2 = box

                    # Convert back to original size
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale

                    # YOLO format
                    x_center = ((x1 + x2) / 2) / original_w
                    y_center = ((y1 + y2) / 2) / original_h

                    width = abs(x2 - x1) / original_w
                    height = abs(y2 - y1) / original_h

                    f.write(
                        f"{CLASS_ID} {x_center} {y_center} {width} {height}\n"
                    )

            print(f"Saved: {label_path}")

            break

        # ==============================
        # SKIP IMAGE
        # ==============================

        elif key == ord('q'):

            break

    cv2.destroyAllWindows()

print("Annotation completed!")