"""Extract objects using N landmarks, or solve spatial-filtering exercises.

COPILOT / EXERCISE (this file also works on its own):
1. Adapt DEFAULT_TEMPLATE: names, normalized targets, canonical_size.
2. Match EVERY click to its named target; never sort the points by position.
3. similarity: 2+ points, shape preserved; affine: 3+ non-collinear points;
   perspective: 4+ coplanar landmarks. ALL points contribute to the fit.
4. Pillow sizes: (width, height). NumPy shapes: (height, width, channels).
5. Clicks use original EXIF-oriented image coordinates, never screen pixels.
6. canonical_size defines framing/aspect ratio; size defines delivered dimensions.
   contain adds padding; cover crops; stretch explicitly distorts the shape.
7. Do not add rotate/crop/resize: source_to_output already composes these steps.
8. For filtering, read FILTER_GUIDE in this file; edit DEFAULT_FILTER.
   Run --task filter without point selection, or --filter-guide for the guide.
The USER places all landmarks manually. Copilot adapts code, never selects points.
This is the only repository file; no external guide or template is required.
Install dependencies: python -m pip install Pillow numpy matplotlib
Manual registration: python main.py photo.jpg --size 300 200
Filtering: python main.py photo.jpg --task filter --filter gaussian
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def validate_size(size: Sequence[int]) -> tuple[int, int]:
    if len(size) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
        for v in size
    ):
        raise ValueError("Expected size: (width, height), two positive integers.")
    return int(size[0]), int(size[1])


def point_array(points, minimum=2) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"At least {minimum} (x, y) points are required.")
    if not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite.")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Two landmarks cannot occupy the same position.")
    return points


@dataclass(frozen=True)
class Template:
    """Ideal landmark positions, normalized within the canonical frame."""
    name: str
    names: tuple[str, ...]
    targets: tuple[tuple[float, float], ...]
    canonical_size: tuple[int, int] = (100, 100)
    method: str = "similarity"

    def __post_init__(self):
        validate_size(self.canonical_size)
        minimum = {"similarity": 2, "affine": 3, "perspective": 4}.get(self.method)
        if minimum is None:
            raise ValueError("Expected method: similarity, affine or perspective.")
        targets = point_array(self.targets, minimum)
        if len(self.names) != len(targets) or len(set(self.names)) != len(self.names):
            raise ValueError("Each target must have a unique name.")
        if any(not isinstance(n, str) or not n.strip() for n in self.names):
            raise ValueError("Each landmark must have a non-empty name.")
        if np.any(targets < 0) or np.any(targets > 1):
            raise ValueError("Normalized targets must be between 0 and 1.")
        if self.method != "similarity" and np.linalg.matrix_rank(targets - targets.mean(axis=0)) < 2:
            raise ValueError("Affine/perspective targets must not be collinear.")

    @property
    def target_pixels(self) -> np.ndarray:
        return np.asarray(self.targets) * np.asarray(self.canonical_size)

    @classmethod
    def from_json(cls, path: str | Path) -> Template:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls(name=data["name"], names=tuple(p["name"] for p in data["landmarks"]),
                       targets=tuple(tuple(p["target"]) for p in data["landmarks"]),
                       canonical_size=tuple(data["canonical_size"]),
                       method=data.get("method", "similarity"))
        except (KeyError, TypeError) as exc:
            raise ValueError("Template needs name, landmarks (name/target), and canonical_size.") from exc


# === EDIT THIS SECTION if only main.py is available during the exercise ===
# Original framing: eyes at 25% and 75% of the width, at 25% of the height.
# For another object, replace BOTH names and targets, and choose its aspect ratio.
DEFAULT_TEMPLATE = Template(
    name="face_two_eyes",
    names=("image_left_eye", "image_right_eye"),
    targets=((0.25, 0.25), (0.75, 0.25)),
    canonical_size=(100, 100),
    method="similarity",
)
# === END OF THE EDITABLE TEMPLATE SECTION ===


def transform_points(matrix: np.ndarray, points) -> np.ndarray:
    """Apply a 3x3 source -> destination matrix, including homogeneous division."""
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("A point is projected to infinity.")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def _normalize(points):
    """Improve numerical stability, especially for large coordinates."""
    center = points.mean(axis=0)
    radius = np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))
    if radius < 1e-10:
        raise ValueError("Landmarks are too close to one another.")
    scale = np.sqrt(2) / radius
    matrix = np.array([[scale, 0, -scale*center[0]],
                       [0, scale, -scale*center[1]], [0, 0, 1]])
    return transform_points(matrix, points), matrix


def estimate_transform(source, target, method="similarity") -> np.ndarray:
    """Fit ALL landmarks globally: source -> canonical frame.

    Similarity and affine: least squares; perspective: normalized DLT.
    Noisy or overdetermined correspondences are fitted approximately.
    """
    minimum = {"similarity": 2, "affine": 3, "perspective": 4}.get(method)
    if minimum is None:
        raise ValueError("Unknown method.")
    source, target = point_array(source, minimum), point_array(target, minimum)
    if source.shape != target.shape:
        raise ValueError("Every source point must have exactly one target.")
    src, src_norm = _normalize(source)
    dst, dst_norm = _normalize(target)
    if method == "similarity":
        # u = a*x - b*y + tx ; v = b*x + a*y + ty : one scale, no reflection.
        design = np.zeros((2*len(src), 4))
        design[0::2, :2] = np.column_stack((src[:, 0], -src[:, 1]))
        design[1::2, :2] = np.column_stack((src[:, 1], src[:, 0]))
        design[0::2, 2] = 1
        design[1::2, 3] = 1
        a, b, tx, ty = np.linalg.lstsq(design, dst.ravel(), rcond=None)[0]
        matrix = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])
    elif method == "affine":
        design = np.column_stack((src, np.ones(len(src))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, dst, rcond=None)
        if rank < 3:
            raise ValueError("An affine transform requires non-collinear points.")
        matrix = np.vstack((coefficients.T, [0, 0, 1]))
    else:
        x, y = src.T
        u, v = dst.T
        one, zero = np.ones(len(src)), np.zeros(len(src))
        design = np.empty((2*len(src), 9))
        design[0::2] = np.column_stack((-x, -y, -one, zero, zero, zero, u*x, u*y, u))
        design[1::2] = np.column_stack((zero, zero, zero, -x, -y, -one, v*x, v*y, v))
        # For an 8x9 matrix (4 points), full_matrices supplies the null-space vector.
        _, singular, vh = np.linalg.svd(design, full_matrices=len(src) == 4)
        if singular[7] < singular[0]*1e-10:
            raise ValueError("Underdetermined perspective: check landmark collinearity.")
        matrix = vh[-1].reshape(3, 3)
    if np.linalg.svd(matrix, compute_uv=False)[-1] < 1e-10:
        raise ValueError("Degenerate transform: check landmarks and their targets.")
    matrix = np.linalg.inv(dst_norm) @ matrix @ src_norm
    return matrix / np.linalg.norm(matrix)


def resize_matrix(canonical_size, output_size, mode="contain") -> np.ndarray:
    """Canonical frame -> output. Equal x/y scales unless stretch is explicit."""
    cw, ch = validate_size(canonical_size)
    ow, oh = validate_size(output_size)
    sx, sy = ow/cw, oh/ch
    if mode == "contain":
        sx = sy = min(sx, sy)
    elif mode == "cover":
        sx = sy = max(sx, sy)
    elif mode != "stretch":
        raise ValueError("Expected resize mode: contain, cover or stretch.")
    return np.array([[sx, 0, (ow-cw*sx)/2], [0, sy, (oh-ch*sy)/2], [0, 0, 1]])


def prepare_image(image: Image.Image) -> Image.Image:
    """Source coordinates always refer to THIS oriented image."""
    image = ImageOps.exif_transpose(image)
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    return image.convert("RGBA" if has_alpha else "RGB")


class SelectionCancelled(Exception):
    pass


def collect_points(image: Image.Image, names: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Left click: add; right click/backspace: undo last; Enter: accept."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    width, height = image.size
    # extent preserves original coordinates even when the window is resized.
    ax.imshow(image, extent=(0, width, height, 0), origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    points, artists = [], []
    accepted = False

    def redraw():
        for artist in artists:
            artist.remove()
        artists.clear()
        for i, (x, y) in enumerate(points):
            artists.extend(ax.plot(x, y, "gx"))
            artists.append(ax.annotate(f"{i+1}: {names[i]}", (x, y), color="lime",
                                       xytext=(6, 6), textcoords="offset points"))
        label = (f"Click {len(points)+1}/{len(names)}: {names[len(points)]}"
                 if len(points) < len(names) else "Press Enter to accept")
        ax.set_title(label + "\nRight click / Backspace: undo | Escape: cancel")
        fig.canvas.draw_idle()

    def on_click(event):
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar and toolbar.mode:
            return
        if event.button == 3 and points:
            points.pop()
        elif event.button == 1 and event.inaxes == ax and len(points) < len(names):
            if event.xdata is None or event.ydata is None:
                return
            point = (float(event.xdata), float(event.ydata))
            if not (0 <= point[0] <= width and 0 <= point[1] <= height) or point in points:
                return
            points.append(point)
        redraw()

    def on_key(event):
        nonlocal accepted
        if event.key in ("backspace", "delete") and points:
            points.pop()
            redraw()
        elif event.key == "enter" and len(points) == len(names):
            accepted = True
            plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)

    ids = [fig.canvas.mpl_connect("button_press_event", on_click),
           fig.canvas.mpl_connect("key_press_event", on_key)]
    redraw()
    try:
        plt.show(block=True)
    finally:
        for connection in ids:
            fig.canvas.mpl_disconnect(connection)
        plt.close(fig)
    if not accepted:
        raise SelectionCancelled("Selection cancelled: no extraction performed.")
    return dict(zip(names, points))


@dataclass
class ExtractionResult:
    image: Image.Image
    metadata: dict


class ObjectExtractor:
    def __init__(self, template=DEFAULT_TEMPLATE, size=None, resize_mode="contain", fill=(0, 0, 0)):
        self.template = template
        self.size = validate_size(template.canonical_size if size is None else size)
        self.resize_mode = resize_mode
        self.layout = resize_matrix(template.canonical_size, self.size, resize_mode)
        if len(fill) != 3 or any(not isinstance(c, int) or not 0 <= c <= 255 for c in fill):
            raise ValueError("Background must contain three RGB integers between 0 and 255.")
        self.fill = tuple(fill)

    def extract(self, img: Image.Image, points=None) -> Image.Image:
        return self.extract_with_metadata(img, points).image

    def extract_with_metadata(self, img: Image.Image, points=None) -> ExtractionResult:
        img = prepare_image(img)
        template = self.template
        if points is None:
            points = collect_points(img, template.names)
        if isinstance(points, Mapping):
            if set(points) != set(template.names):
                raise ValueError(f"Expected landmark names: {', '.join(template.names)}.")
            points = [points[name] for name in template.names]
        source = point_array(points)
        if len(source) != len(template.names):
            raise ValueError(f"{len(template.names)} points required, {len(source)} received.")
        if np.any(source < 0) or np.any(source > np.asarray(img.size)):
            raise ValueError("Points must be inside the original EXIF-oriented image.")
        alignment = estimate_transform(source, template.target_pixels, template.method)
        source_to_output = self.layout @ alignment
        inverse = np.linalg.inv(source_to_output)
        # Reject a projective horizon crossing the requested frame.
        ow, oh = self.size
        corners = np.array([[0, 0, 1], [ow, 0, 1], [ow, oh, 1], [0, oh, 1]])
        denominators = corners @ inverse[2]
        if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
            raise ValueError("The perspective crosses infinity within the requested frame.")
        inverse /= inverse[2, 2]
        fill = self.fill + (255,) if img.mode == "RGBA" else self.fill
        # Pillow needs OUTPUT -> SOURCE; landmarks use the forward matrix.
        if template.method == "perspective":
            kind, coefficients = Image.Transform.PERSPECTIVE, inverse.ravel()[:8]
        else:
            kind, coefficients = Image.Transform.AFFINE, inverse[:2].ravel()
        output = img.transform(self.size, kind, tuple(coefficients),
                               resample=Image.Resampling.BICUBIC, fillcolor=fill)
        if self.resize_mode == "contain":
            # Padding must not contain source pixels outside the canonical frame.
            cw, ch = template.canonical_size
            bounds = transform_points(self.layout, [[0, 0], [cw, ch]])
            left, top, right, bottom = np.ceil(bounds.ravel() - 0.5).astype(int)
            mask = Image.new("L", self.size, 0)
            if right > left and bottom > top:
                ImageDraw.Draw(mask).rectangle((left, top, right-1, bottom-1), fill=255)
            output = Image.composite(output, Image.new(img.mode, self.size, fill), mask)
        actual = transform_points(source_to_output, source)
        expected = transform_points(self.layout, template.target_pixels)
        errors = np.linalg.norm(actual - expected, axis=1)
        metadata = {
            "template": template.name, "method": template.method,
            "source_size": list(img.size), "canonical_size": list(template.canonical_size),
            "output_size": list(self.size), "resize_mode": self.resize_mode,
            "coordinate_system": "x right, y down; edges (0,0)-(width,height); EXIF oriented",
            "source_to_output": source_to_output.tolist(),
            "rms_error_pixels": float(np.sqrt(np.mean(errors**2))),
            "landmarks": [
                {"name": name, "source": src.tolist(), "output": out.tolist(),
                 "target": target.tolist(), "error_pixels": float(error)}
                for name, src, out, target, error in zip(template.names, source, actual, expected, errors)
            ],
        }
        return ExtractionResult(output, metadata)


class FaceExtractor(ObjectExtractor):
    """Compatibility with FaceExtractor(size=(100, 100)) from the original program."""
    def __init__(self, size=(100, 100), resize_mode="contain"):
        super().__init__(DEFAULT_TEMPLATE, size, resize_mode)

    def get_eyes(self, img):
        return list(collect_points(prepare_image(img), self.template.names).values())


# =============================================================================
# FILTERING EXERCISE: SELF-CONTAINED COPILOT GUIDE
# =============================================================================
FILTER_GUIDE = r"""
SPATIAL FILTERING: READ THIS BEFORE ADAPTING THE PROGRAM
======================================================
Source context: Computer Vision - Lecture 5, assessment slide 33 names spatial
filtering; slides 7-9 and 17 discuss smoothing before subsampling and pyramids.
The actual filter is unknown. The examples below are preparation, not a claim
about which filter will be assigned. No lecture file is required at runtime.

COPILOT PROMPT TO FILL IN
-------------------------
Read FILTER_GUIDE, DEFAULT_FILTER, spatial_filter, apply_filter and filter_image
in this same file. Assignment: [paste exact text]. Input: [image/array and range].
Required filter/kernel/formula: [given definition or unknown]. Border policy:
[given or unknown]. Output: [same/valid/full, width x height, gray/RGB].
Adapt DEFAULT_FILTER or the custom kernel first. Keep computation in floating
point; preserve negative responses. Explain convolution versus correlation,
kernel normalization, borders and dimensions. Use the spatial implementation
unless the assignment explicitly requests FFT. Do not invent missing kernel
coefficients. Check the result on a tiny hand-computed example and an impulse.
Keep all code, function documentation and user messages in English.

1. CHOOSE THE OPERATION FROM THE ASSIGNMENT
------------------------------------------
Request                      Existing helper / recipe
Average / mean / box blur    apply_filter(a, 'box', size=(3, 3))
Gaussian smoothing           apply_filter(a, 'gaussian', size=(7, 7), sigma=1.2)
Salt-and-pepper removal      apply_filter(a, 'median', size=(3, 3))
Horizontal derivative Gx     apply_filter(a, 'sobel-x') or 'prewitt-x'
Vertical derivative Gy       apply_filter(a, 'sobel-y') or 'prewitt-y'
Edge magnitude               apply_filter(a, 'sobel') or 'prewitt'
Second derivative           apply_filter(a, 'laplacian')  # signed, zero-sum
Sharpen using Laplacian       apply_filter(a, 'sharpen', amount=1.0)
Unsharp masking              apply_filter(a, 'unsharp', sigma=1.2, amount=1.0)
High-pass detail              apply_filter(a, 'highpass', sigma=1.2)
Local min / max               apply_filter(a, 'minimum') or 'maximum'
Binary threshold             apply_filter(a, 'threshold', threshold=128)
Specified matrix             spatial_filter(a, K, operation='convolution')
Specified sliding dot product spatial_filter(a, K, operation='correlation')

These choices are not interchangeable: median is nonlinear, so there is no
fixed convolution kernel for it. Sobel Gx measures change along x and responds
to vertical edges; Gy measures change down y and responds to horizontal edges.
The supplied Sobel/Prewitt masks are correlation masks: a positive x ramp gives
positive Gx. Their standard coefficients are NOT normalized by default. If the
assignment specifies /8 for Sobel or /6 for Prewitt, divide the returned gradient
components before magnitude/thresholding; multiplying masks changes magnitudes.

2. INPUTS, CHANNELS AND NUMERICAL RANGES
---------------------------------------
Array indexing is a[row, column] = a[y, x]. Shapes are (H,W) or (H,W,C).
Pillow image sizes and --size are (W,H). Kernel size is (ROWS,COLUMNS)=(kh,kw),
not (width,height). E.g. --kernel-size 3 7 smooths more along horizontal x.
image_to_filter_array returns float64, gray or RGB, normally in [0,255]. It
applies EXIF orientation and explicitly composites transparency on --fill.
Low-level array functions preserve the supplied scale: no hidden /255.
Convert uint8 to float BEFORE subtraction, products or convolution; otherwise
negative derivatives wrap around and large sums can overflow. Do not cast to
uint8 until making a display image. Never normalize a numerical result just to
make it look nicer and then reuse that display as the answer. --raw-output
saves the unchanged floating-point result as .npy, BEFORE display mapping/resize.
The spatial engine filters channels independently, never along the channel axis.
For color gradients decide explicitly between grayscale first and per-channel
responses. The CLI defaults to grayscale, matching the lecture practicals.

3. KERNELS: NORMALIZATION, ANCHOR AND CONVOLUTION
------------------------------------------------
A box kernel contains 1/(kh*kw). A sampled Gaussian uses
    exp(-(x*x+y*y)/(2*sigma*sigma))
and divides by its discrete sum. Sigma is in SOURCE pixels; kernel size is its
finite support, not another name for sigma. For odd support, roughly
2*ceil(3*sigma)+1 samples capture most of the Gaussian. A larger support with
the same sigma mainly reduces truncation; it does not necessarily blur more.
Smooth with positive sum-one kernels to preserve constant brightness (except
at zero-padded boundaries). NEVER divide a zero-sum derivative kernel by its
sum. Custom kernels are used exactly as supplied; normalization is explicit.

Correlation slides a patch over the image and sums patch*K. True convolution
uses patch*K[::-1,::-1]. Symmetric blur masks give the same result for both;
asymmetric masks and derivative signs expose the difference. Do not flip twice.
For a tiny manual implementation: pad -> loop output rows/columns -> select the
kh-by-kw patch -> multiply by the effective kernel -> sum into a NEW float array.
Never update the source in place while scanning; that makes results order-dependent.
spatial_filter does exactly this with NumPy views, processing bounded tiles.
Gaussian and box use two separable passes for speed, still in the spatial domain.

4. OUTPUT SIZE AND BORDER POLICY
--------------------------------
For input HxW and kernel khxkw:
  same : H x W                 (DEFAULT; do not resize to repair a wrong shape)
  valid: (H-kh+1) x (W-kw+1)    (kernel must fit; no padded samples)
  full : (H+kh-1) x (W+kw-1)    (all overlap positions, using chosen extension)
For standard mathematical full convolution, choose border='constant', cval=0.
For same, pad before by kh//2, kw//2; after by (kh-1)//2, (kw-1)//2.
With true convolution this equals cropping full at ((kh-1)//2,(kw-1)//2),
matching scipy.signal.convolve2d's centered same result. Even kernels have no
unique central pixel: this convention is explicit, not a license to shift the
result afterward. Correlation here uses the SAME patch alignment as convolution;
other libraries can differ by one pixel for even kernels. Check the assignment.
Odd kernels are easiest, but even custom/Gaussian masks are supported.

Borders are an algorithm parameter, not a cosmetic afterthought:
  reflect   = mirror WITHOUT repeating the edge sample (NumPy convention)
  symmetric = mirror WITH the edge sample repeated
  edge      = repeat the nearest edge value
  constant  = pad with cval (0 by default; can darken a blurred border)
  wrap      = periodic opposite edge
Do not assume scipy.ndimage's mode names equal NumPy's names. Match actual
samples at the boundary. `valid` ignores extension because no padding is used.

5. SIGNED RESPONSES, DISPLAY AND SHARPENING
-------------------------------------------
image_gradients returns (gx,gy,magnitude,angle), with magnitude=hypot(gx,gy)
and angle=atan2(gy,gx) in radians in the image coordinate convention (y down).
The 4-neighbour Laplacian is [[0,1,0],[1,-4,1],[0,1,0]]. It has a negative center.
Sharpening with this convention is image - amount*laplacian(image). Reversing
the Laplacian sign requires reversing that subtraction. Unsharp masking is
image + amount*(image-gaussian(image)); highpass is image-gaussian(image).
Composite filters require same mode to add arrays with matching dimensions.

array_to_image has EXPLICIT display mappings:
  clip      : round/clip [0,255], suitable for blur and sharpened intensities
  normalize : global min/max -> [0,255], flat array -> black (display only)
  signed    : zero -> 128, negative -> dark, positive -> bright (display only)
  absolute  : abs(response)/max(abs(response))*255 (loses sign, display only)
The default is signed for derivatives/highpass, normalize for edge magnitude,
and clip for intensity filters. Save .npy if the grader needs numeric values.
Do not take abs() when reconstructing or adding signed high-pass detail.
Use imshow(raw,cmap='gray',vmin=0,vmax=255) for honest intensity comparisons;
autoscaling each panel independently can hide incorrect brightness changes.

6. FILTER BEFORE DOWNSAMPLING; RESIZE ONLY WHEN REQUESTED
---------------------------------------------------------
A filter normally keeps the original dimensions (same). Standalone filtering
never opens the landmark picker. --size is optional; without it, dimensions
come from the filtering mode. An explicitly requested output resize happens
AFTER filtering/display conversion, with contain/cover/stretch semantics.
--resample lanczos is the default; nearest is available for the lecture demo.
For anti-aliasing, low-pass the original BEFORE subsampling. Blurring the small
aliased output afterward cannot recover the information. Choose sigma/support
from the assignment and reduction factor; there is no universal sigma. A finite
Gaussian attenuates high frequencies, not a perfect brick-wall cutoff.
Example: Gaussian filter, then explicitly downsample to a requested frame:
 python main.py photo.jpg --task filter --filter gaussian --sigma 2 --kernel-size 13 13 --size 160 100 --no-preview
For direct decimation by integer q, blurred[::q,::q] has shape
(ceil(H/q),ceil(W/q)); do not assume exact divisibility. For an exact Wout,Hout,
use the resize helper. --resize contain preserves aspect ratio by adding bars;
cover crops, stretch changes the geometry. NumPy.resize is NOT image resampling.
Filtering after registration is possible through the Python API, but sigma then
means OUTPUT pixels. It does not undo aliasing already introduced by a warp.

7. OTHER LECTURE TOPICS (ONLY IF EXPLICITLY ASKED)
-------------------------------------------------
Gaussian pyramid: repeatedly smooth and reduce; save each actual level shape.
Laplacian pyramid: Li=Gi-expand(Gi+1, exact_size_of_Gi); keep signed floats and
the last coarse Gaussian. Reconstruction adds the expanded coarse level to Li.
For odd dimensions, always resize expansion to the stored previous size.
FFT display: log1p(abs(fftshift(fft2(gray)))). Keep the COMPLEX FFT for processing,
not its log magnitude. Inverse of a shifted spectrum requires ifftshift first.
FFT linear convolution requires padding both arrays to at least
(H+kh-1,W+kw-1), multiplying their FFTs, inverse FFT .real, then a documented crop.
Padding only to image size gives circular convolution. A top-left kernel and a
centered kernel have different phase/alignment: never repair with a guessed roll.
These are study notes; the implemented exercise engine uses spatial filtering.

8. WHAT COPILOT MUST VERIFY
----------------------------
- Identity kernel reproduces input; no channel mixing or uint8 arithmetic.
- A normalized blur preserves a constant away from constant-padded edges.
- An impulse reproduces the expected kernel alignment, including even kernels.
- An asymmetric kernel distinguishes correlation from convolution.
- Horizontal/vertical ramps verify derivative sign and axis; constant -> zero.
- Median removes isolated salt-and-pepper impulses without using a linear mask.
- Portrait and landscape, RGB and gray, same/valid/full shapes are correct.
- Compare tiny arrays with hand calculations, not just a second identical formula.
- Negative results survive until display; raw output remains unchanged.
- Keep manual registration working; no mandatory new dependency.
These are optional manual sanity checks when needed, not a required test suite.
The USER clicks the registration points. Copilot only adapts the code/template.

COMMANDS
--------
 python main.py --filter-guide
 python main.py photo.jpg --task filter --filter median --kernel-size 3 3
 python main.py photo.jpg --task filter --filter sobel --raw-output edges.npy
 python main.py photo.jpg --task filter --filter custom --kernel kernel.json --operation convolution --border constant --filter-mode same
kernel.json contains only the matrix, e.g. [[0,-1,0],[-1,5,-1],[0,-1,0]].
If only this file is allowed, put that matrix directly in DEFAULT_FILTER['kernel'].

References (optional; no network access required):
https://numpy.org/doc/stable/reference/generated/numpy.pad.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
https://pillow.readthedocs.io/en/stable/reference/ImageOps.html
"""

# === EDIT THIS SECTION for a filtering exercise if only main.py is available ===
DEFAULT_FILTER = {
    "name": "gaussian", "size": (5, 5), "sigma": 1.0, "amount": 1.0,
    "threshold": 128.0, "kernel": None, "operation": "convolution",
    "mode": "same", "border": "reflect", "cval": 0.0,
}
FILTER_NAMES = ("identity", "box", "gaussian", "median", "minimum", "maximum",
                "sobel", "sobel-x", "sobel-y", "prewitt", "prewitt-x", "prewitt-y",
                "laplacian", "sharpen", "unsharp", "highpass", "threshold", "custom")
# === END OF THE EDITABLE FILTER SECTION ===


def _float_image(array):
    """Return a finite float64 (H,W) or (H,W,C) array without rescaling it."""
    if np.iscomplexobj(array):
        raise ValueError("Spatial image values must be real, not complex.")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim not in (2, 3) or any(n == 0 for n in array.shape):
        raise ValueError("Expected a non-empty (H,W) or (H,W,C) array.")
    if not np.isfinite(array).all():
        raise ValueError("Image values must be finite.")
    return array


def _kernel_size(size):
    """Kernel dimensions are (rows, columns); a single integer means square."""
    if isinstance(size, (int, np.integer)) and not isinstance(size, (bool, np.bool_)):
        size = (size, size)
    try:
        return validate_size(size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Kernel size must be positive integer (rows, columns).") from exc


def _kernel_array(kernel):
    if np.iscomplexobj(kernel):
        raise ValueError("Kernel coefficients must be real.")
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or 0 in kernel.shape or not np.isfinite(kernel).all():
        raise ValueError("Kernel must be a non-empty finite 2D matrix.")
    return kernel


def box_kernel(size=3):
    """Return a float64 mean kernel of shape (rows,cols), with sum exactly ~1."""
    rows, cols = _kernel_size(size)
    return np.full((rows, cols), 1.0/(rows*cols))


def _gaussian_axis(length, sigma):
    coordinates = np.arange(length, dtype=float) - (length-1)/2
    # Subtract the largest exponent to remain stable for tiny positive sigma.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        squared = coordinates**2
        exponent = -((squared-squared.min()) / sigma) / (2*sigma)
    weights = np.exp(exponent)
    return weights / weights.sum()


def gaussian_kernel(size=5, sigma=1.0):
    """Return a normalized sampled 2D Gaussian; sigma is in source pixels.

    size is (rows,cols) or a square side. Even sizes use half-pixel centered
    samples; output alignment follows spatial_filter's documented convention.
    """
    rows, cols = _kernel_size(size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and greater than zero.")
    return np.outer(_gaussian_axis(rows, sigma), _gaussian_axis(cols, sigma))


def _filter_padding(array, shape, mode, border, cval):
    kh, kw = shape
    h, w = array.shape[:2]
    if border not in ("reflect", "symmetric", "edge", "constant", "wrap"):
        raise ValueError("Unknown border policy.")
    if not np.isfinite(cval):
        raise ValueError("Padding value must be finite.")
    if mode == "same":
        padding = ((kh//2, (kh-1)//2), (kw//2, (kw-1)//2))
    elif mode == "full":
        padding = ((kh-1, kh-1), (kw-1, kw-1))
    elif mode == "valid":
        if kh > h or kw > w:
            raise ValueError("In valid mode, the kernel must fit inside the image.")
        return array
    else:
        raise ValueError("Filter mode must be same, valid or full.")
    if array.ndim == 3:
        padding += ((0, 0),)  # NEVER pad the channel dimension.
    options = {"constant_values": cval} if border == "constant" else {}
    return np.pad(array, padding, mode=border, **options)


def _window_tiles(padded, shape):
    """Yield local window views; bound temporary work even for large median masks."""
    kh, kw = shape
    oh, ow = padded.shape[0]-kh+1, padded.shape[1]-kw+1
    channels = padded.shape[2] if padded.ndim == 3 else 1
    samples_per_window = kh*kw*channels
    cols = min(ow, max(1, 1_000_000 // samples_per_window))
    rows = min(64, max(1, 1_000_000 // (cols*samples_per_window)))
    for y in range(0, oh, rows):
        for x in range(0, ow, cols):
            y1, x1 = min(y+rows, oh), min(x+cols, ow)
            block = padded[y:y1+kh-1, x:x1+kw-1]
            windows = np.lib.stride_tricks.sliding_window_view(block, (kh, kw), axis=(0, 1))
            yield (slice(y, y1), slice(x, x1)), windows


def spatial_filter(array, kernel, *, operation="convolution", mode="same",
                   border="reflect", cval=0.0):
    """Filter gray/RGB/etc. channels independently in the spatial domain.

    Inputs: finite real array (H,W) or (H,W,C), finite kernel (kh,kw).
    Output: float64; same/valid/full shapes are documented in FILTER_GUIDE.
    Convolution flips the kernel; correlation does not. Input is never mutated.
    The even-kernel SAME crop matches scipy.signal.convolve2d for convolution.
    """
    array, kernel = _float_image(array), _kernel_array(kernel)
    if operation == "convolution":
        weights = kernel[::-1, ::-1]
    elif operation == "correlation":
        weights = kernel
    else:
        raise ValueError("Operation must be convolution or correlation.")
    padded = _filter_padding(array, kernel.shape, mode, border, cval)
    shape = (padded.shape[0]-kernel.shape[0]+1, padded.shape[1]-kernel.shape[1]+1)
    output = np.empty(shape + array.shape[2:], dtype=np.float64)
    for region, windows in _window_tiles(padded, kernel.shape):
        output[region] = np.einsum("...ij,ij->...", windows, weights, optimize=False)
    return output


def rank_filter(array, size=3, *, statistic="median", mode="same", border="reflect", cval=0.0):
    """Local median/minimum/maximum, independently per channel, returning float64.

    An even-sized median is the average of the two central sorted samples.
    Minimum/maximum are grayscale erosion/dilation with a rectangular footprint;
    they are not general arbitrary-structuring-element morphology.
    """
    array, shape = _float_image(array), _kernel_size(size)
    reducers = {"median": np.median, "minimum": np.min, "maximum": np.max}
    if statistic not in reducers:
        raise ValueError("Statistic must be median, minimum or maximum.")
    padded = _filter_padding(array, shape, mode, border, cval)
    output = np.empty((padded.shape[0]-shape[0]+1, padded.shape[1]-shape[1]+1)
                      + array.shape[2:], dtype=np.float64)
    for region, windows in _window_tiles(padded, shape):
        output[region] = reducers[statistic](windows, axis=(-2, -1))
    return output


def image_gradients(array, method="sobel", **options):
    """Return signed Gx, Gy, magnitude and atan2(Gy,Gx) radians (y points down).

    Standard unnormalized CORRELATION masks: an increasing x ramp gives positive
    Gx. Sobel yields 8 per unit ramp and Prewitt yields 6 in the interior.
    Pass mode/border/cval as in spatial_filter; do not supply operation.
    """
    if method not in ("sobel", "prewitt"):
        raise ValueError("Gradient method must be sobel or prewitt.")
    middle = 2 if method == "sobel" else 1
    kx = np.array([[-1, 0, 1], [-middle, 0, middle], [-1, 0, 1]], dtype=float)
    gx = spatial_filter(array, kx, operation="correlation", **options)
    gy = spatial_filter(array, kx.T, operation="correlation", **options)
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


def apply_filter(array, name="gaussian", *, size=5, sigma=1.0, amount=1.0,
                 threshold=128.0, kernel=None, operation="convolution",
                 mode="same", border="reflect", cval=0.0):
    """Return the RAW float result. No clipping, display normalization or resize.

    Use custom+kernel for an unknown linear filter. Composite filters (unsharp,
    highpass, sharpen) require same mode so image and response sizes agree.
    Point operations identity/threshold also require same mode.
    """
    array = _float_image(array)
    options = dict(mode=mode, border=border, cval=cval)
    if name not in FILTER_NAMES:
        raise ValueError(f"Unknown filter: {name}.")
    if name in ("identity", "threshold", "unsharp", "highpass", "sharpen") and mode != "same":
        raise ValueError(f"{name} requires same mode; preserve matching image dimensions.")
    if name == "identity":
        return array.copy()
    if name == "threshold":
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite.")
        return np.where(array >= threshold, 255.0, 0.0)
    if name == "custom":
        if kernel is None:
            raise ValueError("Custom filtering requires a kernel matrix.")
        return spatial_filter(array, kernel, operation=operation, **options)
    if name in ("median", "minimum", "maximum"):
        return rank_filter(array, size, statistic=name, **options)
    if name.startswith(("sobel", "prewitt")):
        method = name.split("-")[0]
        gradients = image_gradients(array, method, **options)
        return gradients[0 if name.endswith("-x") else 1 if name.endswith("-y") else 2]
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    if name in ("laplacian", "sharpen"):
        response = spatial_filter(array, laplacian, **options)
        if name == "laplacian":
            return response
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("Sharpening amount must be finite and non-negative.")
        return array - amount*response
    # Separable smoothing: two 1D spatial convolutions, O(H*W*(kh+kw)).
    rows, cols = _kernel_size(size)
    if name == "box":
        ky, kx = np.full((rows, 1), 1/rows), np.full((1, cols), 1/cols)
    else:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian sigma must be finite and greater than zero.")
        ky, kx = _gaussian_axis(rows, sigma)[:, None], _gaussian_axis(cols, sigma)[None, :]
    blurred = spatial_filter(spatial_filter(array, kx, **options), ky, **options)
    if name == "highpass":
        return array - blurred
    if name == "unsharp":
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("Sharpening amount must be finite and non-negative.")
        return array + amount*(array-blurred)
    return blurred


def image_to_filter_array(image, color="gray", fill=(0, 0, 0)):
    """EXIF-orient, composite alpha on RGB fill, then return gray/RGB float64.

    This convenience loader produces 8-bit intensity units [0,255]. For scientific
    or high-bit-depth inputs, load your numeric array and use apply_filter directly.
    """
    image = prepare_image(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, tuple(fill))
        background.paste(image, mask=image.getchannel("A"))
        image = background
    if color not in ("gray", "rgb"):
        raise ValueError("Filter color must be gray or rgb.")
    return np.asarray(image.convert("L" if color == "gray" else "RGB"), dtype=np.float64)


def array_to_image(array, display="clip"):
    """Create an 8-bit gray/RGB display copy. NEVER overwrite the raw response."""
    array = _float_image(array)
    if array.ndim == 3 and array.shape[2] not in (1, 3):
        raise ValueError("Display conversion supports grayscale or three RGB channels.")
    if display == "clip":
        shown = array
    elif display == "normalize":
        low, high = array.min(), array.max()
        shown = (array-low)*255/(high-low) if high > low else np.zeros_like(array)
    elif display in ("signed", "absolute"):
        peak = np.max(np.abs(array))
        normalized = array/peak if peak else np.zeros_like(array)
        shown = 127.5*(normalized+1) if display == "signed" else 255*np.abs(normalized)
    else:
        raise ValueError("Display must be clip, normalize, signed or absolute.")
    shown = np.rint(np.clip(shown, 0, 255)).astype(np.uint8)
    if shown.ndim == 3 and shown.shape[2] == 1:
        shown = shown[..., 0]
    return Image.fromarray(shown)


def resize_filter_output(image, size, mode="contain", resample="lanczos", fill=(0, 0, 0)):
    """Resize a DISPLAY image to exact (width,height), preserving ratio by default.

    This is separate from filter dimensions. For numeric processing retain raw
    values. Low-pass before reduction when the assignment requires anti-aliasing.
    """
    size = validate_size(size)
    methods = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
               "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}
    if resample not in methods:
        raise ValueError("Unknown resampling method.")
    method = methods[resample]
    if mode == "contain":
        background = tuple(fill) if image.mode != "L" else Image.new("RGB", (1, 1), tuple(fill)).convert("L").getpixel((0, 0))
        return ImageOps.pad(image, size, method=method, color=background, centering=(.5, .5))
    if mode == "cover":
        return ImageOps.fit(image, size, method=method, centering=(.5, .5))
    if mode == "stretch":
        return image.resize(size, resample=method)
    raise ValueError("Resize mode must be contain, cover or stretch.")


@dataclass
class FilterResult:
    """Display image, unchanged raw float response, and serializable metadata."""
    image: Image.Image
    raw: np.ndarray
    metadata: dict


def filter_image(image, config=None, *, color="gray", display=None, size=None,
                 resize_mode="contain", resample="lanczos", fill=(0, 0, 0)):
    """Run standalone filtering, without landmark selection. See FilterResult.

    config overrides DEFAULT_FILTER; display and resize affect only result.image.
    result.raw always describes the filter output BEFORE display conversion/resize.
    """
    settings = {**DEFAULT_FILTER, **(config or {})}
    source = image_to_filter_array(image, color, fill)
    raw = apply_filter(source, **settings)
    name = settings["name"]
    if display is None:
        display = ("signed" if name.endswith(("-x", "-y")) or name in ("laplacian", "highpass")
                   else "normalize" if name in ("sobel", "prewitt") else "clip")
    output = array_to_image(raw, display)
    if size is not None:
        output = resize_filter_output(output, size, resize_mode, resample, fill)
    metadata = {
        "task": "filter", "filter": name, "color": color,
        "source_size": [source.shape[1], source.shape[0]],
        "raw_size": [raw.shape[1], raw.shape[0]], "output_size": list(output.size),
        "raw_dtype": str(raw.dtype), "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "display": display, "resize_mode": resize_mode if size is not None else None,
        "resample": resample if size is not None else None,
        "parameters": {k: np.asarray(v).tolist() if isinstance(v, (np.ndarray, tuple)) else v
                       for k, v in settings.items()},
    }
    return FilterResult(output, raw, metadata)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default="1.jpg")
    parser.add_argument("--task", choices=("extract", "filter"), default="extract")
    parser.add_argument("--filter-guide", action="store_true", help="Print the embedded Copilot guide")
    parser.add_argument("--filter", choices=FILTER_NAMES)
    parser.add_argument("--kernel-size", nargs=2, type=int, metavar=("ROWS", "COLS"))
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--amount", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--kernel", type=Path, help="Optional custom kernel JSON matrix")
    parser.add_argument("--operation", choices=("convolution", "correlation"))
    parser.add_argument("--filter-mode", choices=("same", "valid", "full"))
    parser.add_argument("--border", choices=("reflect", "symmetric", "edge", "constant", "wrap"))
    parser.add_argument("--cval", type=float)
    parser.add_argument("--color", choices=("gray", "rgb"), default="gray")
    parser.add_argument("--display", choices=("clip", "normalize", "signed", "absolute"))
    parser.add_argument("--resample", choices=("nearest", "bilinear", "bicubic", "lanczos"), default="lanczos")
    parser.add_argument("--raw-output", type=Path, help="Save the unmodified filter response as .npy")
    parser.add_argument("--template", type=Path, help="JSON template; defaults to DEFAULT_TEMPLATE")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--resize", choices=("contain", "cover", "stretch"), default="contain")
    parser.add_argument("--points", type=Path, help="JSON {name: [x,y]}, oriented image pixels")
    parser.add_argument("--output", type=Path, default=Path("result.png"))
    parser.add_argument("--metadata", type=Path, help="Save matrix, landmarks and errors as JSON")
    parser.add_argument("--fill", nargs=3, type=int, default=(0, 0, 0), metavar=("R", "G", "B"))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args(argv)
    if args.filter_guide:
        print(FILTER_GUIDE)
        return 0
    try:
        if args.filter is not None:
            args.task = "filter"
        if args.raw_output and (args.task != "filter" or args.raw_output.suffix.lower() != ".npy"):
            raise ValueError("--raw-output requires filtering and a .npy filename.")
        if args.task == "filter" and (args.points or args.template):
            raise ValueError("Standalone filtering does not use landmarks or registration templates.")
        destinations = [args.output] + [p for p in (args.metadata, args.raw_output) if p]
        inputs = [Path(args.image)] + [p for p in (args.template, args.points, args.kernel) if p]
        if len({p.resolve() for p in destinations}) != len(destinations) or any(
            out.resolve() == inp.resolve() for out in destinations for inp in inputs
        ):
            raise ValueError("Output paths must differ from input paths and from each other.")
        with Image.open(args.image) as image:
            if args.task == "filter":
                config = {key: value for key, value in {
                    "name": args.filter, "size": args.kernel_size, "sigma": args.sigma,
                    "amount": args.amount, "threshold": args.threshold,
                    "operation": args.operation, "mode": args.filter_mode,
                    "border": args.border, "cval": args.cval,
                }.items() if value is not None}
                if args.kernel:
                    config["kernel"] = json.loads(args.kernel.read_text(encoding="utf-8"))
                result = filter_image(image, config, color=args.color, display=args.display,
                                      size=args.size, resize_mode=args.resize,
                                      resample=args.resample, fill=tuple(args.fill))
                title = result.metadata["filter"]
            else:
                template = Template.from_json(args.template) if args.template else DEFAULT_TEMPLATE
                extractor = ObjectExtractor(template, args.size, args.resize, tuple(args.fill))
                points = json.loads(args.points.read_text(encoding="utf-8")) if args.points else None
                result = extractor.extract_with_metadata(image, points)
                title = template.name
        output = result.image
        if args.output.suffix.lower() in (".jpg", ".jpeg") and output.mode == "RGBA":
            background = Image.new("RGB", output.size, tuple(args.fill))
            background.paste(output, mask=output.getchannel("A"))
            output = background
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.save(args.output)
        if args.raw_output:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            with args.raw_output.open("wb") as stream:
                np.save(stream, result.raw)
        if args.metadata:
            args.metadata.parent.mkdir(parents=True, exist_ok=True)
            args.metadata.write_text(json.dumps(result.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    except SelectionCancelled as exc:
        print(exc)
        return 1
    except (OSError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Image saved: {args.output} ({output.width} × {output.height})")
    if args.task == "extract":
        print(f"Landmark RMS error: {result.metadata['rms_error_pixels']:.3f} output pixels")
    if not args.no_preview:
        import matplotlib.pyplot as plt
        plt.imshow(output, cmap="gray" if output.mode == "L" else None, vmin=0, vmax=255)
        plt.axis("off")
        plt.title(title)
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
