"""
Charts for the ASL project, drawn with NumPy.

Every chart is a plain NumPy image (H x W x 3, uint8, BGR), so it can be shown
with cv2.imshow, saved with cv2.imwrite, or placed next to the camera frame
with np.hstack. Bars and pie slices are filled with NumPy array operations;
OpenCV is used only for text (NumPy has no fonts) and the pie outline.
Nothing beyond numpy and opencv is needed, and both are already in
requirements.txt.

    draw_bar_chart   one horizontal bar per class
    draw_pie_chart   one slice per class
    render_panel     the live panel infer.py shows next to the camera image

Preview without a camera or a trained model:
    python Visualization.py        (c switches bar/pie, q quits)
"""

from functools import lru_cache

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours
BG = (32, 32, 32)
FG = (225, 225, 225)
MUTED = (150, 150, 150)
GRID = (80, 80, 80)
BAR = (170, 130, 80)
ACCEPT = (90, 200, 90)    # top class at or above the confidence threshold
REJECT = (0, 170, 255)    # top class below the confidence threshold

TITLE_H = 28          # pixels reserved for a chart title
AXIS_H = 18           # pixels reserved for the axis labels under the bars
MIN_ROW_H = 13        # smallest bar row that still has readable text
LEGEND_ROW_H = 18

# The live panel is drawn at this size, then scaled to the camera frame's height.
PANEL_W, PANEL_H = 340, 480

ANGLE_BINS = 3600     # angular resolution of the pie: 0.1 degree
_BIN_CENTERS = np.arange(ANGLE_BINS) + 0.5


def _canvas(h, w):
    return np.full((h, w, 3), BG, dtype=np.uint8)


def _text(img, text, org, scale=0.5, color=FG, thickness=1):
    cv2.putText(img, str(text), org, FONT, scale, color, thickness, cv2.LINE_AA)


def _text_width(text, scale):
    return cv2.getTextSize(str(text), FONT, scale, 1)[0][0]


@lru_cache(maxsize=None)
def _palette(n):
    """n distinct BGR colours, evenly spaced around the hue wheel."""
    hsv = np.zeros((1, n, 3), dtype=np.uint8)
    hsv[0, :, 0] = np.arange(n) * 180 // max(n, 1)
    hsv[0, :, 1] = 190
    hsv[0, :, 2] = 230
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]


@lru_cache(maxsize=8)
def _disc_geometry(radius):
    """
    For a disc of this radius: a mask of the pixels inside it, and for each of
    those pixels its angle (0 at 12 o'clock, clockwise) as a bin number.
    This depends only on the radius, so it is computed once and reused.
    """
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    inside = xs * xs + ys * ys <= radius * radius
    angle = np.arctan2(xs, -ys) % (2 * np.pi)
    bins = np.minimum((angle[inside] * (ANGLE_BINS / (2 * np.pi))).astype(np.int32),
                      ANGLE_BINS - 1)
    return inside, bins


def _nice_max(v):
    """Round v up to a tidy axis maximum (for example 1211 -> 1250)."""
    if v <= 0:
        return 1.0
    step = 10 ** np.floor(np.log10(v)) / 4
    return float(np.ceil(v / step) * step)


def draw_bar_chart(values, labels, size, title="", title_color=FG, vmax=None,
                   fmt="{:.0%}", highlight=None, highlight_color=ACCEPT,
                   marker=None, marker_name=""):
    """
    Horizontal bar chart: one bar per class, in the order given.

    values           one number per class
    labels           class names, same order as values
    size             (height, width) in pixels of the returned image
    vmax             value at the right end of the bar area
                     (default: a tidy round number above max(values))
    fmt              format string for the number printed after each bar
    highlight        index of one bar to draw in highlight_color
    marker           value at which to draw a vertical reference line
    marker_name      short text printed above the reference line

    If there are more classes than rows that fit, only the largest are drawn.
    """
    h, w = size
    img = _canvas(h, w)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    vmax = float(vmax) if vmax else _nice_max(values.max() if n else 0.0)

    top, bottom = TITLE_H, h - AXIS_H
    rows_fit = max(1, (bottom - top) // MIN_ROW_H)
    order = np.arange(n) if n <= rows_fit else np.argsort(values)[::-1][:rows_fit]

    heading = title if len(order) == n else f"{title} (top {len(order)})"
    _text(img, heading, (8, 20), 0.55, title_color)
    if n == 0:
        return img

    row_h = (bottom - top) / len(order)
    scale = min(0.55, 0.032 * row_h)
    x0 = min(max(_text_width(labels[i], scale) for i in order) + 14, w // 3)
    x1 = w - _text_width(fmt.format(vmax), scale) - 12
    bar_w = max(x1 - x0, 1)

    for r, i in enumerate(order):
        y0 = top + int(r * row_h)
        y1 = top + int((r + 1) * row_h)
        pad = max(2, int(row_h * 0.18))
        length = int(round(min(values[i] / vmax, 1.0) * bar_w))
        img[y0 + pad:y1 - pad, x0:x0 + length] = highlight_color if i == highlight else BAR
        baseline = (y0 + y1) // 2 + int(11 * scale)
        _text(img, labels[i], (6, baseline), scale)
        if values[i] >= 0.01 * vmax:
            _text(img, fmt.format(values[i]), (x0 + length + 5, baseline), scale,
                  FG if i == highlight else MUTED)

    img[top:bottom, x0 - 1:x0] = GRID                      # baseline of the bars
    for frac in (0.0, 0.5, 1.0):                           # axis labels
        tick = f"{frac * vmax:g}"
        _text(img, tick, (x0 + int(frac * bar_w) - _text_width(tick, 0.4) // 2, h - 5), 0.4, MUTED)

    if marker is not None and 0 < marker <= vmax:
        mx = x0 + int(marker / vmax * bar_w)
        img[top:bottom, mx:mx + 1] = FG
        label = f"{marker_name} {marker:g}".strip()
        _text(img, label, (mx - _text_width(label, 0.4) // 2, TITLE_H - 4), 0.4, FG)
    return img


def draw_pie_chart(values, labels, size, title="", title_color=FG,
                   ring_color=GRID, legend_rows=3, legend_min_share=0.0):
    """
    Pie chart: one slice per class, starting at 12 o'clock and running
    clockwise in the order given.

    The legend lists the largest slices. On a wide canvas it sits beside the
    pie and lists every class; on a narrow one it sits under the pie and
    lists legend_rows entries.

    ring_color        colour of the outline drawn around the pie
    legend_min_share  leave out legend entries whose share is below this
                      (0 lists every class that fits)
    """
    h, w = size
    img = _canvas(h, w)
    _text(img, title, (8, 20), 0.55, title_color)

    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    total = values.sum()
    share = values / total if total > 0 else np.zeros(n)
    colors = _palette(n)
    order = np.argsort(values)[::-1]

    if w >= 1.3 * h:                                       # legend beside the pie
        radius = max((h - TITLE_H - 16) // 2, 8)
        cx, cy = 16 + radius, TITLE_H + 8 + radius
        lx, ly = cx + radius + 30, TITLE_H + 24
        shown = n
    else:                                                  # legend under the pie
        shown = min(legend_rows, n)
        radius = max(min((w - 24) // 2, (h - TITLE_H - shown * LEGEND_ROW_H - 24) // 2), 8)
        cx, cy = w // 2, TITLE_H + 8 + radius
        lx, ly = 12, cy + radius + 22
    shown = min(shown, max(1, (h - ly) // LEGEND_ROW_H))

    if total > 0:
        # Look up which slice each angle bin falls in, then colour every pixel
        # of the disc through its bin. All of it is vectorised NumPy.
        inside, bins = _disc_geometry(radius)
        edges = np.cumsum(share) * ANGLE_BINS              # where each slice ends, in bins
        slice_of_bin = np.minimum(np.searchsorted(edges, _BIN_CENTERS, side="right"), n - 1)
        disc = img[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
        disc[inside] = colors[slice_of_bin[bins]]
    cv2.circle(img, (cx, cy), radius, ring_color, 2, cv2.LINE_AA)

    listed = [i for i in order if share[i] >= legend_min_share][:shown] or [order[0]]
    for r, i in enumerate(listed):
        y = ly + r * LEGEND_ROW_H
        img[y - 10:y + 2, lx:lx + 12] = colors[i]
        _text(img, f"{labels[i]}  {share[i] * 100:.1f}%", (lx + 20, y), 0.5)
    return img


def render_panel(hands, labels, frame_height, mode="bar", threshold=None):
    """
    Live chart panel for infer.py, to be placed to the right of the camera frame.

    hands          list of (title, probabilities, title_color), one per detected hand
    labels         class names in model-output order
    frame_height   height of the camera frame; the panel is scaled to match
    mode           "bar" or "pie"
    threshold      confidence threshold. The top class is green at or above it
                   and orange below it; the bar chart also draws a line at it.
    """
    panel = _canvas(PANEL_H, PANEL_W)
    if not hands:
        _text(panel, "No hand detected", (70, PANEL_H // 2), 0.6, MUTED)
    else:
        slot_h = PANEL_H // len(hands)
        for k, (title, probs, color) in enumerate(hands):
            probs = np.asarray(probs, dtype=np.float64)
            top = int(probs.argmax())
            status = ACCEPT if threshold is None or probs[top] >= threshold else REJECT
            if mode == "pie":
                chart = draw_pie_chart(probs, labels, (slot_h, PANEL_W), title, color,
                                       ring_color=status, legend_min_share=0.005)
            else:
                chart = draw_bar_chart(probs, labels, (slot_h, PANEL_W), title, color,
                                       vmax=1.0, highlight=top, highlight_color=status,
                                       marker=threshold, marker_name="thr")
            panel[k * slot_h:(k + 1) * slot_h] = chart
    panel[:, :2] = GRID                                    # divider next to the camera frame

    if frame_height != PANEL_H:
        width = round(PANEL_W * frame_height / PANEL_H)
        interp = cv2.INTER_AREA if frame_height < PANEL_H else cv2.INTER_LINEAR
        panel = cv2.resize(panel, (width, frame_height), interpolation=interp)
    return panel


if __name__ == "__main__":
    # Quick look at both charts with made-up probabilities (no camera, no model).
    rng = np.random.default_rng(0)
    names = list("ABCDEFGHIKL")

    def fake_probs(peak, sharpness):
        e = np.exp(rng.normal(size=len(names)) + sharpness * (np.arange(len(names)) == peak))
        return e / e.sum()

    demo_hands = [("Right hand", fake_probs(0, 6), (0, 255, 0)),
                  ("Left hand", fake_probs(4, 2), (200, 0, 255))]
    mode = "bar"
    while True:
        cv2.imshow("chart preview", render_panel(demo_hands, names, PANEL_H, mode, threshold=0.8))
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            mode = "pie" if mode == "bar" else "bar"
    cv2.destroyAllWindows()
