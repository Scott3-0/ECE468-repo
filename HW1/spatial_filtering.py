"""ECE468 HW1: spatial filtering of grayscale images.

For an input image ``name.jpg`` these files are written next to it:

* ``nameLaplacian.jpg`` / ``nameLaplacianDiff.jpg``
* ``nameUnsharp.jpg`` / ``nameUnsharpDiff.jpg``
* ``nameAvgLaplace.jpg`` / ``nameAvgLaplaceDiff.jpg``

Usage::

    python spatial_filtering.py image.jpg
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

LAPLACIAN_CENTER_8 = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=np.float64)

LAPLACIAN_CENTER_MINUS_4 = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float64)

AVERAGING_3X3 = np.full((3, 3), 1.0 / 9.0)

JPEG_QUALITY = 95


def gaussian_kernel(size, sigma):
    """Return a normalized ``size`` x ``size`` Gaussian kernel."""
    half = (size - 1) / 2.0
    coords = np.arange(size) - half
    x, y = np.meshgrid(coords, coords)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def filter2d(image, kernel):
    """Correlate a 2-D image with a kernel, replicating border pixels.

    The output has the same shape as the input. All kernels used here
    are symmetric, so correlation and convolution give the same
    result.
    """
    k_rows, k_cols = kernel.shape
    pad_r, pad_c = k_rows // 2, k_cols // 2
    padded = np.pad(image, ((pad_r, pad_r), (pad_c, pad_c)), mode="edge")
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(k_rows):
        for j in range(k_cols):
            result += kernel[i, j] * padded[i:i + rows, j:j + cols]
    return result


def laplacian(image):
    """Apply the 3x3 Laplacian filter with +8 at the center."""
    return filter2d(image, LAPLACIAN_CENTER_8)


def unsharp_mask(image, size=5, sigma=2.0, amount=1.0):
    """Sharpen by adding back the detail removed by a Gaussian blur."""
    blurred = filter2d(image, gaussian_kernel(size, sigma))
    return image + amount * (image - blurred)


def average_then_laplacian(image):
    """Apply the 3x3 averaging filter, then the -4 center Laplacian."""
    averaged = filter2d(image, AVERAGING_3X3)
    return filter2d(averaged, LAPLACIAN_CENTER_MINUS_4)


def to_uint8(image):
    """Clip values to [0, 255] and round to 8-bit integers."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def normalize(image):
    """Linearly rescale values so they span [0, 255]."""
    low, high = image.min(), image.max()
    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)
    return to_uint8((image - low) * 255.0 / (high - low))


def save(image, path):
    """Save an 8-bit grayscale array to ``path``."""
    Image.fromarray(image).save(path, quality=JPEG_QUALITY)


FILTERS = {
    "Laplacian": laplacian,
    "Unsharp": unsharp_mask,
    "AvgLaplace": average_then_laplacian,
}


def spatialFiltering(filename):  # noqa: N802 - name required by assignment
    """Filter ``filename`` with each filter; save results and diffs."""
    path = Path(filename)
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64)

    for filter_type, apply_filter in FILTERS.items():
        filtered = to_uint8(apply_filter(gray))
        save(filtered, path.with_name(f"{path.stem}{filter_type}.jpg"))

        diff = gray - filtered.astype(np.float64)
        save(normalize(diff),
             path.with_name(f"{path.stem}{filter_type}Diff.jpg"))
    return


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filenames", nargs="+", help="input image file(s)")
    args = parser.parse_args()
    for filename in args.filenames:
        spatialFiltering(filename)


if __name__ == "__main__":
    main()
