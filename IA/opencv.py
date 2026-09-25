"""
OpenCV Screenshot / Image Comparison Tool

Features:
- Capture a rectangular region from the screen
- Upload/load PNG, JPG, JPEG, BMP, or WEBP images
- Compare two images using multiple CV metrics
- Automatically resize the second image to the first image's size
- Show:
    * MSE
    * PSNR
    * Mean Absolute Error
    * Pixel difference %
    * Normalized correlation
    * Template matching
    * Edge similarity
    * SSIM
    * Overall similarity score
- Display a visual difference image

Install:
    pip install opencv-python numpy pillow

Run:
    python image_compare.py
"""

import cv2
import numpy as np
import tkinter as tk

from tkinter import filedialog, messagebox
from PIL import ImageGrab, Image, ImageTk


# ============================================================
# Configuration
# ============================================================

DIFFERENCE_THRESHOLD = 30
EDGE_THRESHOLD = 50


# ============================================================
# Image acquisition
# ============================================================

def load_image():
    """Open an image file."""

    path = filedialog.askopenfilename(
        title="Select screenshot",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )

    if not path:
        return None

    image = cv2.imread(path)

    if image is None:
        messagebox.showerror(
            "Error",
            "Could not load the selected image."
        )
        return None

    return image


def capture_screen_region():
    """
    Let the user select a rectangular region of the screen.

    Uses PIL.ImageGrab for the actual screenshot.
    """

    # Take screenshot of the entire screen
    screenshot = ImageGrab.grab()

    # Convert to RGB
    screenshot = screenshot.convert("RGB")

    # Create fullscreen selection window
    selector = tk.Toplevel(root)
    selector.attributes("-fullscreen", True)
    selector.attributes("-topmost", True)
    selector.configure(cursor="crosshair")

    # Convert screenshot for Tkinter
    tk_image = ImageTk.PhotoImage(screenshot)

    canvas = tk.Canvas(
        selector,
        width=screenshot.width,
        height=screenshot.height,
        highlightthickness=0
    )

    canvas.pack(fill="both", expand=True)

    canvas.create_image(
        0,
        0,
        image=tk_image,
        anchor="nw"
    )

    # Important: keep a reference
    canvas.image = tk_image

    selection = {
        "start_x": None,
        "start_y": None,
        "end_x": None,
        "end_y": None,
        "rectangle": None
    }

    def mouse_down(event):
        selection["start_x"] = event.x
        selection["start_y"] = event.y

        if selection["rectangle"]:
            canvas.delete(selection["rectangle"])

        selection["rectangle"] = canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="red",
            width=3
        )

    def mouse_move(event):
        if selection["start_x"] is None:
            return

        canvas.coords(
            selection["rectangle"],
            selection["start_x"],
            selection["start_y"],
            event.x,
            event.y
        )

    def mouse_up(event):
        selection["end_x"] = event.x
        selection["end_y"] = event.y

        selector.destroy()

    canvas.bind("<ButtonPress-1>", mouse_down)
    canvas.bind("<B1-Motion>", mouse_move)
    canvas.bind("<ButtonRelease-1>", mouse_up)

    # ESC cancels
    def cancel(event=None):
        selection["start_x"] = None
        selector.destroy()

    selector.bind("<Escape>", cancel)

    root.wait_window(selector)

    if selection["start_x"] is None:
        return None

    x1 = min(selection["start_x"], selection["end_x"])
    y1 = min(selection["start_y"], selection["end_y"])

    x2 = max(selection["start_x"], selection["end_x"])
    y2 = max(selection["start_y"], selection["end_y"])

    # Prevent zero-size selections
    if x2 <= x1 or y2 <= y1:
        return None

    cropped = screenshot.crop((x1, y1, x2, y2))

    # PIL RGB -> OpenCV BGR
    image = np.array(cropped)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def acquire_image():
    """Ask the user whether to capture or load an image."""

    choice = messagebox.askyesnocancel(
        "Image source",
        "YES = capture a region of the screen\n"
        "NO = load an image file\n"
        "CANCEL = cancel"
    )

    if choice is True:
        return capture_screen_region()

    if choice is False:
        return load_image()

    return None


# ============================================================
# OpenCV metrics
# ============================================================

def mean_squared_error(image1, image2):
    """
    MSE

    Lower is better.

    0 means the images are identical.
    """

    diff = image1.astype(np.float64) - image2.astype(np.float64)

    return np.mean(diff ** 2)


def mean_absolute_error(image1, image2):
    """
    MAE

    Lower is better.
    """

    diff = np.abs(
        image1.astype(np.float64) -
        image2.astype(np.float64)
    )

    return np.mean(diff)


def psnr(image1, image2):
    """
    Peak Signal-to-Noise Ratio.

    Higher is better.

    Identical images return infinity.
    """

    mse = mean_squared_error(image1, image2)

    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    return 10 * np.log10(
        (max_pixel ** 2) / mse
    )


def normalized_correlation(image1, image2):
    """
    Normalized correlation.

    Range is approximately -1 to 1.

    1 = extremely similar
    0 = little correlation
    -1 = opposite relationship
    """

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    a = gray1.astype(np.float64).flatten()
    b = gray2.astype(np.float64).flatten()

    a -= np.mean(a)
    b -= np.mean(b)

    denominator = np.sqrt(
        np.sum(a ** 2) *
        np.sum(b ** 2)
    )

    if denominator == 0:
        return 1.0 if np.array_equal(gray1, gray2) else 0.0

    return np.sum(a * b) / denominator


def pixel_difference_percentage(image1, image2):
    """
    Percentage of pixels that differ significantly.

    Lower is better.
    """

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    _, thresholded = cv2.threshold(
        diff,
        DIFFERENCE_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    different_pixels = np.count_nonzero(thresholded)
    total_pixels = thresholded.size

    return (
        different_pixels /
        total_pixels
    ) * 100


def edge_similarity(image1, image2):
    """
    Compare Canny edge maps.

    Useful for detecting whether shapes, text,
    borders, buttons, etc. are in the same locations.
    """

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    edges1 = cv2.Canny(
        gray1,
        EDGE_THRESHOLD,
        EDGE_THRESHOLD * 3
    )

    edges2 = cv2.Canny(
        gray2,
        EDGE_THRESHOLD,
        EDGE_THRESHOLD * 3
    )

    intersection = np.logical_and(
        edges1 > 0,
        edges2 > 0
    )

    union = np.logical_or(
        edges1 > 0,
        edges2 > 0
    )

    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)

    if union_count == 0:
        return 1.0 if intersection_count == 0 else 0.0

    return intersection_count / union_count


def template_matching_score(image1, image2):
    """
    OpenCV normalized template matching.

    IMPORTANT:
    This assumes both images represent the same region/size.

    Higher is better.
    """

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(
        gray1,
        gray2,
        cv2.TM_CCOEFF_NORMED
    )

    return float(np.max(result))


def ssim(image1, image2):
    """
    Structural Similarity Index.

    Implemented manually so we don't need scikit-image.

    Range:
        0 = very different
        1 = identical

    SSIM considers local brightness, contrast,
    and structure rather than just individual pixels.
    """

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    gray1 = gray1.astype(np.float64)
    gray2 = gray2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(
        gray1,
        (11, 11),
        1.5
    )

    mu2 = cv2.GaussianBlur(
        gray2,
        (11, 11),
        1.5
    )

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(
        gray1 * gray1,
        (11, 11),
        1.5
    ) - mu1_sq

    sigma2_sq = cv2.GaussianBlur(
        gray2 * gray2,
        (11, 11),
        1.5
    ) - mu2_sq

    sigma12 = cv2.GaussianBlur(
        gray1 * gray2,
        (11, 11),
        1.5
    ) - mu1_mu2

    numerator = (
        (2 * mu1_mu2 + C1) *
        (2 * sigma12 + C2)
    )

    denominator = (
        (mu1_sq + mu2_sq + C1) *
        (sigma1_sq + sigma2_sq + C2)
    )

    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


# ============================================================
# Difference visualization
# ============================================================

def create_difference_image(image1, image2):
    """
    Creates a visual heatmap of differences.
    """

    diff = cv2.absdiff(image1, image2)

    # Convert to grayscale
    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY
    )

    # Amplify differences
    amplified = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Apply threshold
    _, thresholded = cv2.threshold(
        amplified,
        DIFFERENCE_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Create red heatmap
    heatmap = cv2.applyColorMap(
        amplified,
        cv2.COLORMAP_JET
    )

    # Highlight significant differences
    heatmap[thresholded == 0] = (0, 0, 0)

    return heatmap


# ============================================================
# Overall score
# ============================================================

def calculate_overall_score(metrics):
    """
    Calculate a normalized 0-100 similarity score.

    This is NOT an OpenCV metric.

    It is a convenience score combining several metrics.

    You should tune the weights for your particular application.
    """

    # SSIM is already 0-1
    ssim_score = np.clip(
        metrics["ssim"],
        0,
        1
    )

    # Correlation -1..1 -> 0..1
    correlation_score = np.clip(
        (metrics["correlation"] + 1) / 2,
        0,
        1
    )

    # Edge similarity 0..1
    edge_score = np.clip(
        metrics["edge_similarity"],
        0,
        1
    )

    # Template matching -1..1 -> 0..1
    template_score = np.clip(
        (metrics["template_matching"] + 1) / 2,
        0,
        1
    )

    # Convert difference percentage to similarity
    pixel_score = np.clip(
        1 - metrics["pixel_difference"] / 100,
        0,
        1
    )

    # Weighted score
    score = (
        ssim_score * 0.35 +
        correlation_score * 0.20 +
        edge_score * 0.15 +
        template_score * 0.15 +
        pixel_score * 0.15
    )

    return score * 100


# ============================================================
# Main comparison
# ============================================================

def compare_images(image1, image2):
    """
    Compare two images using all metrics.
    """

    # Make sure dimensions match
    if image1.shape != image2.shape:

        print(
            f"Resizing second image "
            f"from {image2.shape[1]}x{image2.shape[0]} "
            f"to {image1.shape[1]}x{image1.shape[0]}"
        )

        image2 = cv2.resize(
            image2,
            (
                image1.shape[1],
                image1.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

    metrics = {}

    metrics["mse"] = mean_squared_error(
        image1,
        image2
    )

    metrics["mae"] = mean_absolute_error(
        image1,
        image2
    )

    metrics["psnr"] = psnr(
        image1,
        image2
    )

    metrics["correlation"] = normalized_correlation(
        image1,
        image2
    )

    metrics["pixel_difference"] = pixel_difference_percentage(
        image1,
        image2
    )

    metrics["edge_similarity"] = edge_similarity(
        image1,
        image2
    )

    metrics["template_matching"] = template_matching_score(
        image1,
        image2
    )

    metrics["ssim"] = ssim(
        image1,
        image2
    )

    metrics["overall_score"] = calculate_overall_score(
        metrics
    )

    difference_image = create_difference_image(
        image1,
        image2
    )

    return metrics, difference_image


# ============================================================
# Display results
# ============================================================

def print_results(metrics):
    print("\n" + "=" * 60)
    print("IMAGE COMPARISON RESULTS")
    print("=" * 60)

    print(
        f"MSE                  : "
        f"{metrics['mse']:.4f}"
    )

    print(
        f"MAE                  : "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"PSNR                 : "
        f"{metrics['psnr']:.4f} dB"
    )

    print(
        f"Correlation          : "
        f"{metrics['correlation']:.6f}"
    )

    print(
        f"Pixel difference     : "
        f"{metrics['pixel_difference']:.2f}%"
    )

    print(
        f"Edge similarity      : "
        f"{metrics['edge_similarity']:.6f}"
    )

    print(
        f"Template matching    : "
        f"{metrics['template_matching']:.6f}"
    )

    print(
        f"SSIM                 : "
        f"{metrics['ssim']:.6f}"
    )

    print("-" * 60)

    print(
        f"OVERALL SIMILARITY   : "
        f"{metrics['overall_score']:.2f}%"
    )

    print("=" * 60)


def show_difference(image):
    """Display the difference image in OpenCV."""

    cv2.imshow(
        "Difference Heatmap",
        image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# GUI
# ============================================================

image_a = None
image_b = None


def select_image_a():
    global image_a

    image_a = acquire_image()

    if image_a is not None:
        label_a.config(
            text=(
                f"Image A loaded: "
                f"{image_a.shape[1]} x "
                f"{image_a.shape[0]}"
            )
        )


def select_image_b():
    global image_b

    image_b = acquire_image()

    if image_b is not None:
        label_b.config(
            text=(
                f"Image B loaded: "
                f"{image_b.shape[1]} x "
                f"{image_b.shape[0]}"
            )
        )


def run_comparison():
    if image_a is None:
        messagebox.showwarning(
            "Missing image",
            "Select or capture Image A first."
        )
        return

    if image_b is None:
        messagebox.showwarning(
            "Missing image",
            "Select or capture Image B first."
        )
        return

    metrics, difference = compare_images(
        image_a,
        image_b
    )

    print_results(metrics)

    # Build readable results
    psnr_text = (
        "∞"
        if np.isinf(metrics["psnr"])
        else f"{metrics['psnr']:.2f} dB"
    )

    results = f"""
OVERALL SIMILARITY
{metrics["overall_score"]:.2f}%

--------------------------------

SSIM
{metrics["ssim"]:.6f}

Pixel difference
{metrics["pixel_difference"]:.2f}%

MSE
{metrics["mse"]:.4f}

MAE
{metrics["mae"]:.4f}

PSNR
{psnr_text}

Normalized correlation
{metrics["correlation"]:.6f}

Template matching
{metrics["template_matching"]:.6f}

Edge similarity
{metrics["edge_similarity"]:.6f}
"""

    result_label.config(
        text=results
    )

    show_difference(difference)


# ============================================================
# GUI setup
# ============================================================

root = tk.Tk()

root.title("OpenCV Screenshot Comparison")
root.geometry("600x650")

title = tk.Label(
    root,
    text="OpenCV Image Comparison",
    font=("Arial", 20, "bold")
)

title.pack(pady=20)


label_a = tk.Label(
    root,
    text="Image A: Not selected",
    font=("Arial", 12)
)

label_a.pack(pady=5)


button_a = tk.Button(
    root,
    text="Select / Capture Image A",
    command=select_image_a,
    width=35,
    height=2
)

button_a.pack(pady=10)


label_b = tk.Label(
    root,
    text="Image B: Not selected",
    font=("Arial", 12)
)

label_b.pack(pady=5)


button_b = tk.Button(
    root,
    text="Select / Capture Image B",
    command=select_image_b,
    width=35,
    height=2
)

button_b.pack(pady=10)


compare_button = tk.Button(
    root,
    text="COMPARE IMAGES",
    command=run_comparison,
    width=35,
    height=3,
    bg="#1976D2",
    fg="white",
    font=("Arial", 12, "bold")
)

compare_button.pack(pady=20)


result_label = tk.Label(
    root,
    text="Results will appear here.",
    font=("Courier New", 11),
    justify="left"
)

result_label.pack(
    padx=20,
    pady=10
)


root.mainloop()
