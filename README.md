# Object extraction with 2, 3, 4... N landmarks

Click landmarks on an image, align the object to a reference template, and save an image with the exact requested dimensions. The engine stays in **`main.py`**, which also works on its own with Pillow, NumPy and Matplotlib. The original two-eye face framing remains the default template.

## Quick start

Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python main.py photo.jpg --size 300 200 --output result.png
```

If only `main.py` is available: `python -m pip install Pillow numpy matplotlib`.
Without arguments, the program opens `1.jpg`. No input photograph is included.

1. Click the landmark named in the window title, in the requested order.
2. Right-click or press Backspace to undo the last point.
3. After placing every point, press **Enter**. Escape or closing the window cancels.
4. The result is saved and displayed. `--no-preview` disables only the final display.

Matplotlib zoom and pan do not record points: turn the toolbar tool off to resume selection. Clicking requires a graphical environment. The `--points` option below also works without a GUI.

## Choose an object template

```bash
python main.py photo.jpg --template examples/face_5_points.json --size 300 300
python main.py photo.jpg --template examples/object_3_points.json --size 300 500
python main.py photo.jpg --template examples/card_4_corners.json --size 900 600
```

These templates are **geometry examples**, not object detectors. Adapt their positions and proportions to the assignment. The five face positions are illustrative. The card uses a 3:2 aspect ratio; change it for A4 paper or a differently proportioned card.

| Method | Landmarks | Effect and use |
| --- | --- | --- |
| `similarity` | 2 or more distinct points | Rotation, translation and uniform scale. Default for preserving shape. All points contribute to the fit. |
| `affine` | 3 or more non-collinear points | Also allows shear, stretching and possibly reflection. Use only when the correction requires this. See `object_3_points_affine.json`. |
| `perspective` | 4 or more points in a non-degenerate arrangement | Rectifies a **flat** object such as a card seen at an angle. With four corners, no three may be collinear. |

More points do not automatically require a different method: five face landmarks can still use `similarity`. There is no hard-coded two-point limit.

The canonical frame defines the intended crop, including margins. Each target is a fraction of this frame: `[0.25, 0.40]` means 25% of its width and 40% of its height. Each source point matches a target **with the same name**, never an automatic left-to-right sorting.

## Dimensions and resizing

`--size WIDTH HEIGHT` specifies the **exact** saved dimensions. The template's aspect ratio comes from `canonical_size`, independently of the input photo's dimensions.

| Option | When output and template aspect ratios differ |
| --- | --- |
| `--resize contain` (default) | Keeps the entire frame and its proportions; adds background padding. |
| `--resize cover` | Fills the output without stretching by cropping part of the frame. Some landmarks may fall outside the image. |
| `--resize stretch` | Fills the output by scaling width and height independently. Requires an explicit choice. |

Example: square template to 300x200 with `contain` gives centered 200x200 content and 50-pixel side bars. The file is exactly 300x200. For a naturally rectangular object, first define a suitable rectangular template.

```bash
python main.py photo.jpg --size 300 200 --resize contain --fill 255 255 255
```

Rotation, framing and final scale are composed into one matrix and sampled directly from the original with bicubic interpolation. No intermediate rotation can clip the object, and no small canonical raster is enlarged afterward. Canonical dimensions describe virtual geometry. Very strong downsampling of fine textures may benefit from specialized prefiltering; this engine does not reconstruct missing details.

When the requested frame extends beyond the source photo, missing pixels use `--fill`. Shape is preserved by `similarity` with `contain`/`cover`; `affine` and `perspective` deliberately change geometry before resizing. Source transparency is retained in PNG and flattened against the background for JPEG output.

## Replay points and inspect results

Create `points.json` using coordinates in the **original image after EXIF orientation**:

```json
{
  "image_left_eye": [120, 90],
  "image_right_eye": [240, 110]
}
```

```bash
python main.py photo.jpg --points points.json --size 300 200 --metadata result.json --no-preview
```

The output JSON records the oriented source size, canonical and output sizes, the source-to-output matrix, each transformed landmark and its target error. RMS error is measured in **output pixels**. It can reveal incorrect correspondences or unsuitable geometry, but zero error does not prove that the landmarks have the correct semantic meaning. Outliers are not automatically discarded.

## Python API

```python
from PIL import Image
from main import ObjectExtractor, Template

template = Template.from_json("examples/face_5_points.json")
extractor = ObjectExtractor(template, size=(320, 240), resize_mode="contain")
with Image.open("photo.jpg") as image:
    result = extractor.extract_with_metadata(image)  # select all five landmarks
result.image.save("result.png")
print(result.metadata["rms_error_pixels"])
```

`extract(image, points=...)` returns only the image. `points` accepts a dictionary of names or a list in the exact template order. `FaceExtractor(size=(100, 100)).extract(image)` remains available. To reproduce the original program's stretching into a non-square output, explicitly use `resize_mode="stretch"`.

## Adapt during the exercise

Read **[ADAPT_WITH_COPILOT.md](ADAPT_WITH_COPILOT.md)** for a step-by-step method, dimension formulas, pitfalls and a ready-to-copy prompt. Repository instructions live in **[.github/copilot-instructions.md](.github/copilot-instructions.md)**, using the [repository instruction format documented by GitHub](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions-in-your-ide/add-repository-instructions-in-your-ide). If only `main.py` is accessible, its header and `DEFAULT_TEMPLATE` contain the essential rules.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover transforms, the contribution of additional points, rectangular/odd dimensions, padding, EXIF orientation, transparency, selection/undo/cancellation and headless saving. CI runs the suite on Linux and Windows.
