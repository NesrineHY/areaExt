import cv2
import numpy as np
from pathlib import Path

# ==========================
# Configuration
# ==========================

INPUT_DIR = "examples"
OUTPUT_DIR = "Results/straightFloodFillResults"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

MIN_AREA = 7000  # Ignore tiny contours

extensions = ("*.jpg", "*.jpeg", "*.png")

# ==========================
# Process images
# ==========================

for ext in extensions:
    for img_path in Path(INPUT_DIR).glob(ext):

        print(f"Processing {img_path.name}")

        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binary image (black objects become white)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )

        # ------------------------------------------------
        # Isolate long straight lines (box borders), discard text blobs
        # ------------------------------------------------

        # Long thin kernels: only survive if a run of white pixels is at least this long
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
        vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

        lines_only = cv2.bitwise_or(horiz_lines, vert_lines)

        # Reconnect small gaps between line segments (corners, dashed lines)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(lines_only, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # ------------------------------------------------
        # Flood fill from top-left (outside)
        # ------------------------------------------------

        flood = closed.copy()

        h, w = flood.shape

        # FloodFill requires a mask 2 pixels larger
        mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(flood, mask, (0, 0), 255)

        # Background mask
        background = flood

        # Everything NOT connected to the outside
        foreground = cv2.bitwise_not(background)

        # ------------------------------------------------
        # Contours
        # ------------------------------------------------

        contours, hierarchy = cv2.findContours(
            foreground,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        result = img.copy()

        # Filter contours by area first, then draw each with a distinct color
        filtered = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
        for idx, c in enumerate(filtered):
            # Approximate contour with straight line segments (preserves concave/L shapes)
            epsilon = 0.01 * cv2.arcLength(c, True)   # tweak 0.005–0.02 to taste
            c = cv2.approxPolyDP(c, epsilon, True)

            hsv_color = np.uint8([[[int((idx * 180) / max(1, len(filtered))), 255, 255]]])
            bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            cv2.drawContours(result, [c], -1, color, 5)

        cv2.imwrite(
            str(Path(OUTPUT_DIR) / img_path.name),
            result,
        )

print("Done.")