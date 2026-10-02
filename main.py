"""Extract an object using N landmarks.

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
See also .github/copilot-instructions.md and ADAPT_WITH_COPILOT.md.
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
            raise ValueError("Invalid template: see examples/*.json.") from exc


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default="1.jpg")
    parser.add_argument("--template", type=Path, help="JSON template; defaults to DEFAULT_TEMPLATE")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--resize", choices=("contain", "cover", "stretch"), default="contain")
    parser.add_argument("--points", type=Path, help="JSON {name: [x,y]}, oriented image pixels")
    parser.add_argument("--output", type=Path, default=Path("result.png"))
    parser.add_argument("--metadata", type=Path, help="Save matrix, landmarks and errors as JSON")
    parser.add_argument("--fill", nargs=3, type=int, default=(0, 0, 0), metavar=("R", "G", "B"))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args(argv)
    try:
        destinations = [args.output] + ([args.metadata] if args.metadata else [])
        inputs = [Path(args.image)] + [p for p in (args.template, args.points) if p]
        if len({p.resolve() for p in destinations}) != len(destinations) or any(
            out.resolve() == inp.resolve() for out in destinations for inp in inputs
        ):
            raise ValueError("Output paths must differ from input paths and from each other.")
        template = Template.from_json(args.template) if args.template else DEFAULT_TEMPLATE
        extractor = ObjectExtractor(template, args.size, args.resize, tuple(args.fill))
        points = json.loads(args.points.read_text(encoding="utf-8")) if args.points else None
        with Image.open(args.image) as image:
            result = extractor.extract_with_metadata(image, points)
        output = result.image
        if args.output.suffix.lower() in (".jpg", ".jpeg") and output.mode == "RGBA":
            background = Image.new("RGB", output.size, tuple(args.fill))
            background.paste(output, mask=output.getchannel("A"))
            output = background
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.save(args.output)
        if args.metadata:
            args.metadata.parent.mkdir(parents=True, exist_ok=True)
            args.metadata.write_text(json.dumps(result.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    except SelectionCancelled as exc:
        print(exc)
        return 1
    except (OSError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(f"Image saved: {args.output} ({output.width} × {output.height})")
    print(f"Landmark RMS error: {result.metadata['rms_error_pixels']:.3f} output pixels")
    if not args.no_preview:
        import matplotlib.pyplot as plt
        plt.imshow(output)
        plt.axis("off")
        plt.title(template.name)
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
