# Adapting the program with GitHub Copilot

## Prompt to copy at the start of the exercise

```text
Read .github/copilot-instructions.md, ADAPT_WITH_COPILOT.md and main.py before editing.
If only main.py is available, read its entire header and DEFAULT_TEMPLATE.

Exact assignment: [PASTE THE ASSIGNMENT]
Object: [OBJECT TYPE]
Landmarks to click: [ORDERED LIST OF NAMES]
Final dimensions: [WIDTH] x [HEIGHT] pixels.
Framing, margin and proportion constraints: [CONSTRAINTS OR UNKNOWN]

Adapt a JSON template or DEFAULT_TEMPLATE first. Keep the generic N-point engine.
Each click must match its named target. Use ALL points, including points 3, 4, 5
and beyond; do not reduce the selection to the first two.
Explain the choice of similarity / affine / perspective from the object's geometry.
Ask for essential missing information instead of inventing physical proportions.
Separate canonical_size (reference aspect ratio/framing) from size (exact file size).
Preserve proportions with contain unless the assignment explicitly requires otherwise.
Check landscape/portrait outputs and the final landmark coordinates.
Run available tests, then give me the command and exact click order.
Use English throughout.
```

Open the **repository root** in the IDE and ensure repository instructions are enabled. If Copilot misses them, explicitly attach the three files to the chat. If the exercise only permits `main.py`, take that file: it contains the engine, default template and essential invariants.

## 1. Describe the object's ideal frame

Keep three coordinate spaces separate:

| Space | Coordinates | Example |
| --- | --- | --- |
| Source | Pixels of the EXIF-oriented photo, regardless of display size | Click `(1234.5,567.2)` inside a 4000x3000 photo |
| Canonical | Ideal frame with chosen aspect ratio and margins | Vertical object in a 300x500 frame |
| Output | Pixels of the delivered file | 240x240, with padding if needed |

Targets are normalized inside the canonical frame. For example:

```python
DEFAULT_TEMPLATE = Template(
    name="vertical_object",
    names=("tip", "left_base", "right_base"),
    targets=((0.50, 0.10), (0.15, 0.85), (0.85, 0.85)),
    canonical_size=(300, 500),
    method="similarity",
)
```

Targets become `(150,50)`, `(45,425)` and `(255,425)` in canonical pixels. These are the desired positions, **not coordinates copied from the input photo**. To use the equivalent JSON, run `python main.py photo.jpg --template examples/object_3_points.json --size 240 240`.

Two points determine orientation and scale, but cannot reveal the object's unknown height or margins. Derive them from the assignment or specify them explicitly in the template. The original face framing places eyes at `(0.25,0.25)` and `(0.75,0.25)` in a square: the crop side is twice the eye separation.

## 2. Use more than two points correctly

To move from 2 to N, supply N names and N targets. The picker automatically requests N clicks. Do not introduce `left, right = ...` or a two-point-only loop into the engine.

With `similarity`, one transformation is fitted to **all** correspondences:

```text
u = a*x - b*y + tx
v = b*x + a*y + ty
```

The 2N equations determine four unknowns `(a,b,tx,ty)` by least squares. Scale is `sqrt(a*a+b*b)`. With three or more landmarks, the fit balances residuals. A misplaced nose or third vertex therefore really affects the result. Correct inaccurate clicks and choose realistic targets.

Do not add source landmarks without corresponding targets. Do not independently fit pairs and average their angles. Do not automatically choose affine because there are three points or perspective because there are four.

For a flat object photographed at an angle, `perspective` can place its four corners in a rectangle. Order refers to the rectified object: top left -> top right -> bottom right -> bottom left. Do not reconstruct that order by sorting photo coordinates. With more than four coplanar landmarks, every additional point also needs a known target; normalized DLT fits them jointly.

A single global transform cannot unfold a 3D object or deformed fabric. An assignment requiring contour segmentation, local warping or measurement alone needs a specific extension; the picker and dimension conventions remain reusable.

## 3. Choose and check final resizing

Notation: canonical frame `(cw,ch)`, output `(ow,oh)`. To preserve proportions:

```text
contain: s = min(ow/cw, oh/ch)
cover:   s = max(ow/cw, oh/ch)
dx = (ow - s*cw)/2
dy = (oh - s*ch)/2
x_output = s*x_canonical + dx
y_output = s*y_canonical + dy
```

Example: 300x500 template, 240x240 output, `contain`: s=0.48, content 144x240, dx=48, dy=0. The canonical tip `(150,50)` lands at `(120,24)`. Side bars are 48 pixels each. The saved image is exactly 240x240.

For the same output with `cover`, s=0.8, content 240x400, dy=-80: top and bottom are cropped. With `stretch`, sx=0.8 and sy=0.48: the object is distorted.

Changing output resolution does not change the object's shape. For a 3:2 card, keep a 600x400 template even if the required file is square. Conversely, if the object's real aspect ratio changes, update canonical geometry. Four corners alone cannot tell perspective correction the correct physical aspect ratio.

`source_to_output` already includes resizing. Pass its inverse to Pillow and request `size`. Another `.resize(...)` after extraction would invalidate exported point coordinates and could distort the image. `canonical_size` is not an intermediate raster: increasing only `size` samples the original directly at the required resolution.

## 4. Pixel conventions and common mistakes

- `image.size == (width,height)`; `np.asarray(image).shape == (height,width,channels)`.
- x is horizontal; y increases downward. NumPy indexes `[row,column]`, meaning `[y,x]`.
- Frame edges run from `(0,0)` to `(width,height)`; the first pixel center is `(0.5,0.5)`.
- The Matplotlib window may be small; `event.xdata/ydata` stays in original coordinates thanks to `extent`. `event.x/y` gives screen pixels and must not drive geometry.
- Apply EXIF orientation before clicking. A points JSON must use the same orientation.
- Keep coordinates and matrices as floating-point numbers. Round only where a raster API requires it.
- Pillow samples with the **inverse** transform (output -> source). Landmarks and metadata use the **forward** transform (source -> output).
- Do not crop an original after `rotate(expand=False)`. The engine directly samples every useful source pixel.
- When framing extends beyond the photo, retain the requested dimensions and fill missing pixels; do not shrink the crop by clamping its bounds.

## 5. Verify an adaptation

```bash
python -m unittest discover -s tests -v
python main.py photo.jpg --template examples/object_3_points.json --size 300 200 --metadata result.json
```

Check the actual saved dimensions, not just the window. Verify landmark order, margins, aspect ratio, additional points and portrait output. Metadata reports target and actual positions: with `similarity` and N>2, a small residual is normal when the object differs from the template. Large errors suggest inaccurate clicks, unsuitable template proportions or an inappropriate method. No universal error threshold is imposed.

Technical references: [Pillow conventions](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system), [Pillow inverse matrices](https://pillow.readthedocs.io/en/stable/reference/ImageTransform.html), [Matplotlib display coordinates](https://matplotlib.org/stable/users/explain/artists/imshow_extent.html).
