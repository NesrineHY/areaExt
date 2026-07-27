import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = "examples"
OUTPUT_DIR = "Results/combined"
MIN_AREA = 20000
MIN_DIMENSION = 20
FLOOD_MIN_AREA = 7000
OVERLAP_IOU_THRESHOLD = 0.35

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_color_boxes(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_low_sat = cv2.inRange(hsv, (0, 15, 170), (180, 150, 255))
    mask_blue = cv2.inRange(hsv, (85, 20, 150), (135, 160, 255))
    mask = cv2.bitwise_or(mask_low_sat, mask_blue)

    text_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(text_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            continue

        boxes.append((x, y, w, h, "color", None))

    return boxes


def get_flood_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    flood = closed.copy()
    h, w = flood.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)

    foreground = cv2.bitwise_not(flood)
    contours, _ = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < FLOOD_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            continue

        boxes.append((x, y, w, h, "flood", cnt))

    return boxes


def box_area(box):
    return box[2] * box[3]


def intersection_over_union(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = box_area(box_a) + box_area(box_b) - inter_area
    return inter_area / union_area


def merge_boxes(boxes):
    kept = []
    for box in sorted(boxes, key=box_area, reverse=True):
        overlapping = [existing for existing in kept if intersection_over_union(box, existing) >= OVERLAP_IOU_THRESHOLD]
        if not overlapping:
            kept.append(box)
            continue

        if box[4] == "color":
            continue

        kept = [existing for existing in kept if existing not in overlapping]
        kept.append(box)

    return kept


for img_path in sorted(Path(INPUT_DIR).glob("*")):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    color_boxes = get_color_boxes(img)
    flood_boxes = get_flood_boxes(img)
    all_boxes = color_boxes + flood_boxes
    final_boxes = merge_boxes(all_boxes)

    result = img.copy()
    flood_boxes = [box for box in final_boxes if box[4] == "flood"]
    for idx, box in enumerate(flood_boxes):
        _, _, _, _, _, contour = box
        hsv_color = np.uint8([[[int((idx * 180) / max(1, len(flood_boxes))), 255, 255]]])
        bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        cv2.drawContours(result, [contour], -1, color, 5)

    for idx, (x, y, w, h, method, _) in enumerate(final_boxes):
        if method == "flood":
            continue

        color = (0, 0, 255)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        cv2.putText(result, str(idx + 1), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(str(Path(OUTPUT_DIR) / img_path.name), result)
    print(f"{img_path.name}: {len(final_boxes)} final boxes")

print("Done.")
