# Repository instructions

Use English for code, comments, documentation, messages and explanations. This program supports an exercise: click N landmarks on a photo, align/frame an object, and deliver an image with specified dimensions. The object type can change. Keep `main.py` standalone with Pillow, NumPy and Matplotlib; do not add OpenCV or an unnecessary module framework.

## Adaptation workflow

First understand the assignment: landmark names/count, source-target correspondences, reference geometry, margins, final width and height. Ask for missing essential geometry rather than inventing physical proportions. Prefer editing a JSON template in `examples/`, or `DEFAULT_TEMPLATE` if only `main.py` is permitted. Do not rewrite the generic engine for each object.

## Rules for N points

- `Template.names` and `Template.targets` must each contain N items matched by index. Names are unique. A normalized target `(u,v)` becomes `(u*cw,v*ch)` inside `canonical_size=(cw,ch)`.
- Adding a landmark requires both a name and a target in the same order. `collect_points` automatically adapts the number of clicks and labels.
- Use every point in `estimate_transform`. Never use `points[:2]`, choose an arbitrary pair, average pairwise angles or replace correspondences with a bounding box.
- Never sort points by x/y: this breaks correspondence on rotated objects. Corner order is top left, top right, bottom right, bottom left **in the intended object orientation**. For anatomical landmarks, distinguish image-left from the person's left.
- Use `similarity` for 2+ distinct landmarks: rotation, translation, uniform scale, no reflection. More than two points remain valid and jointly minimize the fitting error.
- Use `affine` only when affine deformation is intended: 3+ non-collinear landmarks. Use `perspective` only for a plane: 4+ landmarks in a non-degenerate arrangement; with four corners, no three may be collinear. Point count alone never determines the method.
- Do not promise exact alignment of every point under a similarity if their geometry differs from the template. Keep per-point and RMS errors. Correct outliers rather than silently dropping them.

## Coordinate and dimension invariants

- Pillow and `--size`: `(width,height)`; `np.asarray(image).shape`: `(height,width,channels)`. x increases rightward, y downward.
- Call `prepare_image` BEFORE selection. Coordinates refer to the original EXIF-oriented image. `imshow(..., extent=(0,w,h,0))` and `event.xdata/ydata` preserve this system under zoom/window resizing. Never use `event.x/y` or resize the source for display.
- Convention: image edges `(0,0)` to `(w,h)`, first pixel center `(0.5,0.5)`. Retain floating-point coordinates until rasterization; do not round at every stage.
- `canonical_size` defines the reference aspect ratio/frame, with margins encoded by targets. `size` defines exact delivered dimensions. Do not automatically replace `canonical_size` with `size`.
- `resize_matrix` computes R: `contain` uses s=min(ow/cw,oh/ch); `cover` uses s=max(...); dx=(ow-s*cw)/2, dy=(oh-s*ch)/2. `stretch` uses sx=ow/cw, sy=oh/ch only when explicitly requested.
- `source_to_output = R @ alignment`. Transform landmark coordinates with this forward matrix. Pillow `Image.transform` needs its INVERSE: output -> source. Never swap these uses.
- Render directly at `size` with bicubic interpolation and a single geometric transform. Do not add `rotate(expand=False)`, an intermediate crop or a second `.resize(size)`. The `contain` mask creates actual background bars.
- Do not clamp crop bounds to the source: that would change framing/aspect ratio. Out-of-source pixels use `fill`.
- `contain` preserves the frame with padding; `cover` can clip the object; `stretch` distorts it. Affine/perspective may already change geometry before resizing: do not promise globally unchanged proportions in those modes.

## Validation after adaptation

Run `python -m unittest discover -s tests -v`.
Check a two-point case, a 3+ point case where the last point changes the fit, 300x200 and 200x300 outputs, margins, correcting clicks, and framing beyond the source. For perspective, test a skewed quadrilateral rather than only identity. Verify `result.image.size == (width,height)` and exported landmark positions. Do not describe automated checks as manual testing.

Explain the chosen template, click order, exact command and resize mode. `ADAPT_WITH_COPILOT.md` supplies examples and a prompt to fill in.
