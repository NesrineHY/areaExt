"""
Docstrum page-layout analysis
==============================
Implementation of Lawrence O'Gorman's "The Document Spectrum for Page
Layout Analysis" (IEEE TPAMI, Vol 15, No 11, Nov 1993).

Pipeline (mirrors the paper's Section III):
  1. Preprocessing        -> binarize + light denoise
  2. Connected components -> centroids + bounding boxes
  3. k-NN clustering      -> Dij(d, phi) for each component  (the "docstrum")
  4. Orientation estimate -> circular histogram of angles
  5. Spacing estimate     -> within-line / between-line histograms
  6. Text-line formation  -> transitive closure + linear regression
  7. Block formation      -> parallelism / proximity / overlap grouping

Author: implementation for interactive use, not affiliated with the
original paper's author.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------
def binarize(gray, method="otsu"):
    """Return a boolean array where True = ink (foreground)."""
    from skimage.filters import threshold_otsu, threshold_sauvola

    gray = gray.astype(np.float64)
    if method == "otsu":
        t = threshold_otsu(gray)
    else:
        t = threshold_sauvola(gray, window_size=25)
    # Assume text is darker than background
    return gray < t


def denoise(binary, min_size=2):
    """Light salt-and-pepper cleanup (playing the role of kFill in the
    paper): remove foreground specks smaller than min_size pixels and
    fill background holes smaller than min_size pixels."""
    from skimage.morphology import remove_small_objects, remove_small_holes

    cleaned = remove_small_objects(binary, min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=min_size)
    return cleaned


# ----------------------------------------------------------------------
# 2. Connected components
# ----------------------------------------------------------------------
def connected_components(binary):
    """
    Label connected components and return a structured array of features:
    centroid (y, x), bounding box (min_row, min_col, max_row, max_col),
    and size (sqrt of bbox area, as used in the paper's Fig. 4 histogram).
    """
    labeled, n = ndi.label(binary, structure=np.ones((3, 3)))
    objs = ndi.find_objects(labeled)

    centroids = ndi.center_of_mass(binary, labeled, index=np.arange(1, n + 1))
    centroids = np.array(centroids)  # (y, x)

    boxes = []
    sizes = []
    for sl in objs:
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        boxes.append((y0, x0, y1, x1))
        sizes.append(np.sqrt((y1 - y0) * (x1 - x0)))
    boxes = np.array(boxes)
    sizes = np.array(sizes)

    return {
        "labeled": labeled,
        "n": n,
        "centroids": centroids,        # (n, 2) -> (y, x)
        "boxes": boxes,                # (n, 4) -> (y0, x0, y1, x1)
        "sizes": sizes,                # (n,)
    }


def filter_by_size(comps, low=None, high=None):
    """Keep only components whose size falls in [low, high]. If bounds are
    None, they are estimated automatically from the size histogram
    (peak-based heuristic per the paper's Section III-B)."""
    sizes = comps["sizes"]
    if low is None:
        low = 3.0
    if high is None:
        # crude peak-based heuristic: histogram peak * 3 (paper's default)
        hist, edges = np.histogram(sizes, bins=50)
        # ignore the first couple of bins (often noise) when finding the peak
        peak_bin = np.argmax(hist[2:]) + 2
        peak_size = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
        high = peak_size * 3.0

    mask = (sizes >= low) & (sizes <= high)
    idx = np.where(mask)[0]

    filtered = {
        "centroids": comps["centroids"][idx],
        "boxes": comps["boxes"][idx],
        "sizes": comps["sizes"][idx],
        "orig_index": idx,
    }
    return filtered, (low, high)


# ----------------------------------------------------------------------
# 3. k-NN clustering -> docstrum
# ----------------------------------------------------------------------
def k_nearest_neighbors(centroids, k=5):
    """
    centroids: (n,2) array of (y, x).
    Returns pairs (i, j), distance d, and angle phi in [0, 180) degrees,
    where phi is measured counterclockwise from horizontal (x axis) as in
    the paper. Image y grows downward, so we flip the sign of dy to get a
    conventional (mathematical, upward-positive) angle.
    """
    n = centroids.shape[0]
    tree = cKDTree(centroids)
    # query k+1 because the first neighbor returned is the point itself
    dists, idxs = tree.query(centroids, k=k + 1)

    pairs_i = []
    pairs_j = []
    pairs_d = []
    pairs_phi = []

    for i in range(n):
        for rank in range(1, k + 1):
            j = idxs[i, rank]
            d = dists[i, rank]
            dy = centroids[i, 0] - centroids[j, 0]   # y_i - y_j
            dx = centroids[j, 1] - centroids[i, 1]   # x_j - x_i
            phi = np.degrees(np.arctan2(dy, dx))     # math convention, y-up
            phi = phi % 180.0                        # undirected -> [0,180)
            pairs_i.append(i)
            pairs_j.append(j)
            pairs_d.append(d)
            pairs_phi.append(phi)

    return {
        "i": np.array(pairs_i),
        "j": np.array(pairs_j),
        "d": np.array(pairs_d),
        "phi": np.array(pairs_phi),
    }


# ----------------------------------------------------------------------
# 4. Orientation estimate (circular histogram of angles)
# ----------------------------------------------------------------------
def estimate_orientation(phi, bin_deg=0.5, smooth_frac=0.25):
    """
    phi: array of angles in [0, 180).
    Returns the orientation estimate in degrees, in (-90, 90], measuring
    the dominant text-line direction.
    """
    n_bins = int(round(180.0 / bin_deg))
    hist, edges = np.histogram(phi, bins=n_bins, range=(0, 180))

    window_len = max(1, int(round(n_bins * smooth_frac)))
    if window_len % 2 == 0:
        window_len += 1
    kernel = np.ones(window_len) / window_len

    # circular smoothing: wrap the histogram around before convolving
    padded = np.concatenate([hist[-window_len:], hist, hist[:window_len]])
    smoothed = np.convolve(padded, kernel, mode="same")
    smoothed = smoothed[window_len:window_len + n_bins]

    peak_bin = np.argmax(smoothed)
    peak_angle = edges[peak_bin] + bin_deg / 2.0  # in [0, 180)

    # convert to (-90, 90]
    if peak_angle > 90:
        peak_angle -= 180
    return peak_angle, hist, edges


# ----------------------------------------------------------------------
# 5. Spacing estimates (within-line / between-line histograms)
# ----------------------------------------------------------------------
def _hist_peak(values, bin_size, smooth_window):
    if len(values) == 0:
        return None, None, None
    vmin, vmax = values.min(), values.max()
    n_bins = max(1, int(round((vmax - vmin) / bin_size)))
    hist, edges = np.histogram(values, bins=n_bins, range=(vmin, vmax))

    w = max(1, smooth_window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    smoothed = np.convolve(hist, kernel, mode="same")

    peak_bin = np.argmax(smoothed)
    peak_val = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
    return peak_val, hist, edges


def estimate_spacing(d, phi, orientation_deg, angle_tol=30.0,
                      bin_px=2.0, smooth_window=5):
    """
    Split nearest-neighbor pairs into "within-line" (angle close to the
    orientation) and "between-line" (angle close to orientation + 90) sets,
    then find the peak spacing (distance) for each.
    """
    orient = orientation_deg % 180.0
    perp = (orient + 90.0) % 180.0

    def ang_dist(a, b):
        diff = np.abs(a - b) % 180.0
        return np.minimum(diff, 180.0 - diff)

    within_mask = ang_dist(phi, orient) <= angle_tol
    between_mask = ang_dist(phi, perp) <= angle_tol

    within_d = d[within_mask]
    between_d = d[between_mask]

    within_peak, wh, we = _hist_peak(within_d, bin_px, smooth_window)
    between_peak, bh, be = _hist_peak(between_d, bin_px, smooth_window)

    return {
        "within_line_spacing": within_peak,
        "between_line_spacing": between_peak,
        "within_mask": within_mask,
        "between_mask": between_mask,
        "within_hist": (wh, we),
        "between_hist": (bh, be),
    }


# ----------------------------------------------------------------------
# 6. Text-line formation
# ----------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_text_lines(centroids, nn, within_mask, max_dist=None):
    """
    Transitive closure on within-line nearest-neighbor pairings, then a
    least-squares line fit to centroids of each resulting group.
    Returns a list of dicts: {indices, p0, p1, angle_deg}.
    """
    n = centroids.shape[0]
    uf = UnionFind(n)

    ii, jj, dd = nn["i"][within_mask], nn["j"][within_mask], nn["d"][within_mask]
    if max_dist is not None:
        keep = dd <= max_dist
        ii, jj = ii[keep], jj[keep]

    for a, b in zip(ii, jj):
        uf.union(a, b)

    groups = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    lines = []
    for root, members in groups.items():
        pts = centroids[members]  # (m, 2) -> (y, x)
        if len(members) == 1:
            y, x = pts[0]
            lines.append({
                "indices": members,
                "p0": (y, x), "p1": (y, x),
                "angle_deg": None,
            })
            continue

        xs = pts[:, 1]
        ys = pts[:, 0]
        # Fit line minimizing perpendicular distance via total least squares
        # (PCA on centered points) -- more robust than ordinary y=mx+b
        # when lines are near-vertical.
        mean = pts.mean(axis=0)
        centered = pts - mean
        # SVD gives principal direction
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]  # (dy, dx) unit vector of the line
        # project points onto direction to get extent -> endpoints
        proj = centered @ direction
        p_min = mean + direction * proj.min()
        p_max = mean + direction * proj.max()
        angle = np.degrees(np.arctan2(-direction[0], direction[1])) % 180

        lines.append({
            "indices": members,
            "p0": tuple(p_min), "p1": tuple(p_max),
            "angle_deg": angle,
        })

    return lines


# ----------------------------------------------------------------------
# 7. Block formation
# ----------------------------------------------------------------------
def _line_overlap_perp(line_a, line_b):
    """
    Approximate parallel overlap and perpendicular distance between two
    text-line segments, following the paper's Section III-F (simplified
    with vector projections instead of the closed-form algebra given
    there -- mathematically equivalent).
    """
    p0a, p1a = np.array(line_a["p0"]), np.array(line_a["p1"])
    p0b, p1b = np.array(line_b["p0"]), np.array(line_b["p1"])

    dir_a = p1a - p0a
    len_a = np.linalg.norm(dir_a)
    if len_a == 0:
        return 0.0, np.linalg.norm(p0a - p0b)
    dir_a_unit = dir_a / len_a

    # project b's endpoints onto a's direction
    proj0 = np.dot(p0b - p0a, dir_a_unit)
    proj1 = np.dot(p1b - p0a, dir_a_unit)
    seg_b = sorted([proj0, proj1])
    seg_a = [0.0, len_a]

    overlap = min(seg_a[1], seg_b[1]) - max(seg_a[0], seg_b[0])

    # perpendicular distance: distance from midpoint of b to line a
    perp_dir = np.array([-dir_a_unit[1], dir_a_unit[0]])
    mid_b = (p0b + p1b) / 2.0
    perp_dist = abs(np.dot(mid_b - p0a, perp_dir))

    return overlap, perp_dist


def form_blocks(lines, between_line_spacing, within_line_spacing,
                 angle_tol=30.0, perp_factor=1.3, parallel_factor=1.5):
    """
    Group text lines into structural blocks based on parallelism,
    perpendicular proximity, and (parallel) overlap/proximity, per the
    paper's Section III-F.
    """
    n = len(lines)
    uf = UnionFind(n)

    max_perp = perp_factor * (between_line_spacing or 1.0)
    max_parallel_gap = parallel_factor * (within_line_spacing or 1.0)

    angles = [ln["angle_deg"] if ln["angle_deg"] is not None else 0.0
              for ln in lines]

    def ang_dist(a, b):
        diff = abs(a - b) % 180.0
        return min(diff, 180.0 - diff)

    for a in range(n):
        for b in range(a + 1, n):
            if ang_dist(angles[a], angles[b]) > angle_tol:
                continue
            overlap, perp = _line_overlap_perp(lines[a], lines[b])
            if perp > max_perp:
                continue
            if overlap < 0 and abs(overlap) > max_parallel_gap:
                continue
            uf.union(a, b)

    groups = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)

    blocks = []
    for root, members in groups.items():
        all_pts = []
        for m in members:
            all_pts.append(lines[m]["p0"])
            all_pts.append(lines[m]["p1"])
        pts = np.array(all_pts)
        y0, x0 = pts[:, 0].min(), pts[:, 1].min()
        y1, x1 = pts[:, 0].max(), pts[:, 1].max()
        blocks.append({
            "line_indices": members,
            "bbox": (y0, x0, y1, x1),  # (min_row, min_col, max_row, max_col)
        })

    return blocks


# ----------------------------------------------------------------------
# Top-level convenience function
# ----------------------------------------------------------------------
def run_docstrum(binary_image, k=5, size_low=None, size_high=None,
                  angle_tol=30.0, perp_factor=1.3, parallel_factor=1.5,
                  verbose=True):
    """
    Run the full docstrum pipeline on a preprocessed boolean image
    (True = ink). Returns a dict with every intermediate result so you can
    inspect / plot each stage.
    """
    comps = connected_components(binary_image)
    if verbose:
        print(f"Connected components found: {comps['n']}")

    filtered, size_bounds = filter_by_size(comps, size_low, size_high)
    if verbose:
        print(f"Kept {len(filtered['sizes'])} components after size "
              f"filter {size_bounds}")

    centroids = filtered["centroids"]
    nn = k_nearest_neighbors(centroids, k=k)

    orientation, angle_hist, angle_edges = estimate_orientation(nn["phi"])
    if verbose:
        print(f"Preliminary orientation estimate: {orientation:.2f} deg")

    spacing = estimate_spacing(nn["d"], nn["phi"], orientation,
                                angle_tol=angle_tol)
    if verbose:
        print(f"Within-line spacing: {spacing['within_line_spacing']}")
        print(f"Between-line spacing: {spacing['between_line_spacing']}")

    max_within_dist = None
    if spacing["within_line_spacing"] is not None:
        max_within_dist = 3.0 * spacing["within_line_spacing"]

    lines = find_text_lines(centroids, nn, spacing["within_mask"],
                             max_dist=max_within_dist)
    if verbose:
        print(f"Text lines found: {len(lines)}")

    valid_angles = [ln["angle_deg"] for ln in lines if ln["angle_deg"] is not None]
    if valid_angles:
        final_orientation, _, _ = estimate_orientation(
            np.array(valid_angles) % 180.0)
    else:
        final_orientation = orientation
    if verbose:
        print(f"Final (refined) orientation: {final_orientation:.2f} deg")

    blocks = form_blocks(lines,
                          spacing["between_line_spacing"],
                          spacing["within_line_spacing"],
                          angle_tol=angle_tol,
                          perp_factor=perp_factor,
                          parallel_factor=parallel_factor)
    if verbose:
        print(f"Structural blocks found: {len(blocks)}")

    return {
        "components": comps,
        "filtered": filtered,
        "nn": nn,
        "orientation_prelim": orientation,
        "orientation_final": final_orientation,
        "spacing": spacing,
        "lines": lines,
        "blocks": blocks,
    }
