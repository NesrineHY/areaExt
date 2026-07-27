"""Color segmentation: detect homogeneous shaded regions, reject lines."""
import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = "examples"
OUTPUT_DIR = "Results/colorSeg"
MIN_AREA = 20000
MIN_DIMENSION = 20  # reject if either w or h < this (lines are thin)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for img_path in sorted(Path(INPUT_DIR).glob("*")):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Light colored regions: low saturation + high value
    mask_low_sat = cv2.inRange(hsv, (0, 15, 170), (180, 150, 255))
    # Light blue range
    mask_blue = cv2.inRange(hsv, (85, 20, 150), (135, 160, 255))
    mask = cv2.bitwise_or(mask_low_sat, mask_blue)

    # Remove dark text pixels
    text_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(text_mask))

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Reject lines: too thin in either dimension
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            continue

        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)
        count += 1

    cv2.imwrite(str(Path(OUTPUT_DIR) / img_path.name), result)
    print(f"{img_path.name}: {count} regions")

print("Done.")
