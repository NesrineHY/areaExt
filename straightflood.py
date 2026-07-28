import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi

# Import your existing docstrum module
from docstrum import connected_components, filter_by_size

# ==========================
# Configuration
# ==========================

INPUT_DIR = "examples"
OUTPUT_DIR = "Results/straightFloodFillResults"
DEBUG_DIR = "Results/debug"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)

MIN_AREA = 7000
APPROX_EPSILON_FRAC = 0.01

# Widen or narrow the "this is text" band found by the docstrum peak heuristic.
# >1.0 removes more (catches slightly bigger/smaller letters too).
TEXT_LOW_MULT = 0.5
TEXT_HIGH_MULT = 1.0   # filter_by_size's own `high` already = peak * 3, keep as-is or tune

extensions = ("*.jpg", "*.jpeg", "*.png")


def remove_text_components(binary):
    """
    binary: boolean or 0/255 uint8 array, True/255 = ink.
    Uses docstrum's connected-component size histogram to find the
    dominant "letter size" cluster, then erases those components,
    leaving lines/borders/larger graphics intact.
    """
    bool_binary = binary > 0

    comps = connected_components(bool_binary)
    filtered, (low, high) = filter_by_size(comps)  # docstrum's peak-based band

    # Widen/narrow the band a bit if needed
    low = low * TEXT_LOW_MULT
    high = high * TEXT_HIGH_MULT

    labeled = comps["labeled"]
    sizes = comps["sizes"]  # indexed 0..n-1, corresponds to label i+1

    # Which *original* component labels fall inside the text-size band?
    text_label_ids = np.where((sizes >= low) & (sizes <= high))[0] + 1  # labels are 1-indexed

    cleaned = binary.copy()
    text_mask = np.isin(labeled, text_label_ids)
    cleaned[text_mask] = 0

    return cleaned, text_mask, (low, high)


# ==========================
# Process images
# ==========================

for ext in extensions:
    for img_path in Path(INPUT_DIR).glob(ext):

        print(f"Processing {img_path.name}")

        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binary image (dark strokes/lines become white)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )

        # ------------------------------------------------
        # Remove letter-sized connected components (docstrum size filter)
        # ------------------------------------------------
        no_text, text_mask, bounds = remove_text_components(binary)
        print(f"  text-size band used: {bounds}")

        cv2.imwrite(str(Path(DEBUG_DIR) / f"textmask_{img_path.name}"),
                    (text_mask * 255).astype(np.uint8))
        cv2.imwrite(str(Path(DEBUG_DIR) / f"notext_{img_path.name}"), no_text)

        # ------------------------------------------------
        # Close small gaps left where borders touched removed text
        # ------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(no_text, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ------------------------------------------------
        # Flood fill from top-left (outside)
        # ------------------------------------------------
        flood = closed.copy()
        h, w = flood.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)

        background = flood
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
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

        for idx, c in enumerate(filtered_contours):
            epsilon = APPROX_EPSILON_FRAC * cv2.arcLength(c, True)
            c_approx = cv2.approxPolyDP(c, epsilon, True)

            hsv_color = np.uint8([[[int((idx * 180) / max(1, len(filtered_contours))), 255, 255]]])
            bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

            cv2.drawContours(result, [c_approx], -1, color, 5)

        cv2.imwrite(str(Path(OUTPUT_DIR) / img_path.name), result)

print("Done.")