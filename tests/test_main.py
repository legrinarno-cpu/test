import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image

from main import (DEFAULT_TEMPLATE, FaceExtractor, ObjectExtractor, SelectionCancelled,
                  Template, collect_points, estimate_transform, main, prepare_image,
                  resize_matrix, transform_points, validate_size)

ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_similarity_two_three_five_and_many_points(self):
        rng = np.random.default_rng(12)
        for count in (2, 3, 5, 20, 1000):
            with self.subTest(count=count):
                src = rng.uniform(-100, 900, (count, 2))
                # 90° clockwise in image coordinates, scale 2, translation.
                expected = np.column_stack((700 - 2*src[:, 1], 50 + 2*src[:, 0]))
                matrix = estimate_transform(src, expected)
                assert_allclose(transform_points(matrix, src), expected, atol=1e-9)
                linear = matrix[:2, :2] / matrix[2, 2]
                assert_allclose(linear.T @ linear, np.eye(2)*4, atol=1e-9)
                self.assertGreater(np.linalg.det(linear), 0)

    def test_additional_points_change_the_fit(self):
        src = np.array([[10, 10], [90, 10], [50, 50], [20, 80], [80, 80.]])
        target = src.copy()
        target[-1] += [15, -20]
        all_points = estimate_transform(src, target)
        two_points = estimate_transform(src[:2], target[:2])
        all_error = np.sum((transform_points(all_points, src) - target)**2)
        two_error = np.sum((transform_points(two_points, src) - target)**2)
        self.assertLess(all_error, two_error)
        self.assertFalse(np.allclose(transform_points(all_points, src), src))

    def test_affine_uses_more_than_three_points(self):
        src = np.array([[0, 0], [100, 0], [0, 100], [90, 80], [40, 20.]])
        target = src @ np.array([[2, .3], [.2, 1.5]]) + [30, -40]
        matrix = estimate_transform(src, target, "affine")
        assert_allclose(transform_points(matrix, src), target, atol=1e-10)
        target[-1] += [5, 10]
        matrix = estimate_transform(src, target, "affine")
        first_three = estimate_transform(src[:3], target[:3], "affine")
        self.assertLess(np.sum((transform_points(matrix, src)-target)**2),
                        np.sum((transform_points(first_three, src)-target)**2))

    def test_perspective_four_and_more_points(self):
        src = np.array([[20, 20], [180, 10], [160, 150], [10, 130], [70, 60], [110, 90.]])
        # Independent analytic projection, including nonzero projective terms.
        denominator = 1 + .001*src[:, 0] + .002*src[:, 1]
        target = np.column_stack(((1.2*src[:, 0]+.1*src[:, 1]+5)/denominator,
                                  (-.2*src[:, 0]+1.5*src[:, 1]+10)/denominator))
        for count in (4, 6):
            with self.subTest(count=count):
                matrix = estimate_transform(src[:count], target[:count], "perspective")
                assert_allclose(transform_points(matrix, src), target, atol=1e-8)

    def test_large_coordinates_are_normalized(self):
        src = np.array([[1e8, 2e8], [1e8+1000, 2e8], [1e8+50, 2e8+800]])
        target = (src-[1e8, 2e8]) * .5 + 10
        for method in ("similarity", "affine"):
            with self.subTest(method=method):
                assert_allclose(transform_points(estimate_transform(src, target, method), src),
                                target, atol=1e-6)

    def test_extra_perspective_landmark_is_not_ignored(self):
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [60, 40.]])
        dst = src.copy()
        dst[-1] += [10, -5]
        fitted = transform_points(estimate_transform(src, dst, "perspective"), src)
        self.assertFalse(np.allclose(fitted, src))
        self.assertLess(np.sum((fitted-dst)**2), np.sum((src-dst)**2))

    def test_degenerate_or_invalid_points_are_rejected(self):
        cases = [([], [], "similarity"), ([[1, 1]]*2, [[0, 0], [1, 1]], "similarity"),
                 ([[0, 0], [float("nan"), 1]], [[0, 0], [1, 1]], "similarity"),
                 ([[0, 0], [1, 1]], [[0, 0], [1, 1]], "affine"),
                 ([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 0], [0, 1]], "affine"),
                 ([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 1], [2, 2]], "affine"),
                 ([[0, 0], [1, 0], [2, 0], [3, 0]],
                  [[0, 0], [1, 0], [1, 1], [0, 1]], "perspective")]
        for src, dst, method in cases:
            with self.subTest(src=src, method=method), self.assertRaises(ValueError):
                estimate_transform(src, dst, method)

    def test_resize_matrices_and_aspect_ratio(self):
        # 100×100 -> 300×150: contain=150×150 centred, cover=300×300 cropped.
        assert_allclose(resize_matrix((100, 100), (300, 150)),
                        [[1.5, 0, 75], [0, 1.5, 0], [0, 0, 1]])
        assert_allclose(resize_matrix((100, 100), (300, 150), "cover"),
                        [[3, 0, 0], [0, 3, -75], [0, 0, 1]])
        assert_allclose(resize_matrix((100, 100), (300, 150), "stretch"),
                        [[3, 0, 0], [0, 1.5, 0], [0, 0, 1]])

    def test_invalid_dimensions(self):
        for size in ((0, 10), (-1, 5), (10.5, 5), (True, 10), (10,), (10, 20, 30)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                validate_size(size)


class ExtractionTests(unittest.TestCase):
    def test_identity_preserves_pixels_and_landmarks(self):
        rng = np.random.default_rng(5)
        pixels = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        result = ObjectExtractor().extract_with_metadata(Image.fromarray(pixels), [[25, 25], [75, 25]])
        # Bicubic floating point roundoff can change channels by one level.
        self.assertLessEqual(np.max(np.abs(np.asarray(result.image).astype(int)-pixels)), 1)
        self.assertLess(result.metadata["rms_error_pixels"], 1e-9)
        assert_allclose([p["output"] for p in result.metadata["landmarks"]], [[25, 25], [75, 25]])

    def test_rotation_does_not_clip_source_before_cropping(self):
        # Tall original: the old rotate(expand=False) would clip the long red part.
        pixels = np.zeros((200, 40, 3), dtype=np.uint8)
        pixels[150:180, 10:30] = [255, 0, 0]
        template = Template("bar", ("a", "b"), ((.1, .5), (.9, .5)), (200, 40))
        output = ObjectExtractor(template).extract(Image.fromarray(pixels), [[20, 20], [20, 180]])
        self.assertEqual(output.size, (200, 40))
        self.assertGreater(np.asarray(output)[20, 160, 0], 250)

    def test_contain_bands_are_background_even_if_source_has_more_pixels(self):
        image = Image.new("RGB", (500, 300), "white")
        result = ObjectExtractor(size=(300, 150), fill=(12, 34, 56)).extract_with_metadata(
            image, [[125, 125], [175, 125]])
        self.assertEqual(result.image.size, (300, 150))
        self.assertEqual(result.image.getpixel((0, 50)), (12, 34, 56))
        self.assertEqual(result.image.getpixel((74, 50)), (12, 34, 56))
        self.assertEqual(result.image.getpixel((75, 50)), (255, 255, 255))
        self.assertEqual(result.image.getpixel((224, 50)), (255, 255, 255))
        self.assertEqual(result.image.getpixel((225, 50)), (12, 34, 56))
        assert_allclose([p["output"] for p in result.metadata["landmarks"]],
                        [[112.5, 37.5], [187.5, 37.5]], atol=1e-9)

    def test_exact_odd_portrait_landscape_and_small_dimensions(self):
        image = Image.new("RGB", (100, 100), "white")
        for size in ((301, 149), (149, 301), (1, 1), (1, 17), (17, 1)):
            for mode in ("contain", "cover", "stretch"):
                with self.subTest(size=size, mode=mode):
                    result = ObjectExtractor(size=size, resize_mode=mode).extract(image, [[25, 25], [75, 25]])
                    self.assertEqual(result.size, size)

    def test_crop_beyond_source_is_filled(self):
        result = ObjectExtractor(fill=(7, 8, 9)).extract(Image.new("RGB", (100, 100), "white"),
                                                      [[0, 0], [50, 0]])
        self.assertEqual(result.getpixel((0, 0)), (7, 8, 9))
        self.assertEqual(result.getpixel((50, 50)), (255, 255, 255))

    def test_named_points_are_reordered_by_template_not_dict(self):
        image = Image.new("RGB", (100, 100))
        result = ObjectExtractor().extract_with_metadata(image,
            {"image_right_eye": [75, 25], "image_left_eye": [25, 25]})
        self.assertLess(result.metadata["rms_error_pixels"], 1e-9)
        self.assertEqual(result.metadata["landmarks"][0]["source"], [25, 25])

    def test_points_count_names_and_bounds(self):
        image = Image.new("RGB", (100, 100))
        for points in ({"a": [25, 25], "b": [75, 25]}, [[-1, 0], [10, 10]],
                       [[10, 101], [20, 20]], [[10, 10], [20, 20], [30, 30]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                ObjectExtractor().extract(image, points)

    def test_exif_orientation_is_applied_before_points(self):
        image = Image.new("RGB", (120, 60), "white")
        image.getexif()[274] = 6
        self.assertEqual(prepare_image(image).size, (60, 120))
        result = ObjectExtractor().extract_with_metadata(image, [[10, 90], [40, 90]])
        self.assertEqual(result.metadata["source_size"], [60, 120])

    def test_image_modes_and_transparency(self):
        for mode in ("L", "RGB", "RGBA", "P", "CMYK"):
            with self.subTest(mode=mode):
                image = Image.new(mode, (100, 100))
                output = FaceExtractor().extract(image, [[25, 25], [75, 25]])
                self.assertEqual(output.mode, "RGBA" if mode == "RGBA" else "RGB")
        image = Image.new("P", (100, 100))
        image.info["transparency"] = 0
        output = ObjectExtractor().extract(image, [[25, 25], [75, 25]])
        self.assertEqual(output.getpixel((50, 50))[3], 0)

    def test_perspective_raster_uses_inverse_mapping(self):
        yy, xx = np.mgrid[:200, :200]
        pixels = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.uint8)
        template = Template("card", ("tl", "tr", "br", "bl"),
                            ((0, 0), (1, 0), (1, 1), (0, 1)), (100, 80), "perspective")
        result = ObjectExtractor(template).extract_with_metadata(Image.fromarray(pixels),
                                                                 [[20, 10], [170, 25], [155, 180], [40, 160]])
        inverse = np.linalg.inv(np.array(result.metadata["source_to_output"]))
        for x, y in ((15, 20), (50, 40), (80, 60)):
            expected = transform_points(inverse, [[x+.5, y+.5]])[0] - .5
            assert_allclose(result.image.getpixel((x, y))[:2], expected, atol=2)

    def test_examples_load_and_have_valid_geometry(self):
        for path in (ROOT / "examples").glob("*.json"):
            with self.subTest(path=path):
                template = Template.from_json(path)
                image = Image.new("RGB", template.canonical_size, "white")
                result = ObjectExtractor(template).extract_with_metadata(image, template.target_pixels)
                self.assertLess(result.metadata["rms_error_pixels"], 1e-8)

    def test_crossed_perspective_corners_are_rejected(self):
        template = Template("card", ("tl", "tr", "br", "bl"),
                            ((0, 0), (1, 0), (1, 1), (0, 1)), (100, 100), "perspective")
        with self.assertRaisesRegex(ValueError, "infinity"):
            ObjectExtractor(template).extract(Image.new("RGB", (100, 100)),
                                               [[10, 10], [90, 90], [90, 10], [10, 90]])


class PickerTests(unittest.TestCase):
    def run_picker(self, actions, names=("a", "b", "c")):
        def fake_show(**kwargs):
            fig = plt.gcf()
            fig.set_size_inches(4, 3)
            fig.canvas.draw()
            ax = fig.axes[0]
            for kind, value in actions:
                if kind == "key":
                    event = KeyEvent("key_press_event", fig.canvas, key=value)
                elif kind == "close":
                    plt.close(fig)
                    break
                else:
                    x, y = ax.transData.transform(value)
                    event = MouseEvent("button_press_event", fig.canvas, x, y,
                                       button=1 if kind == "click" else 3)
                fig.canvas.callbacks.process(event.name, event)
        with patch("matplotlib.pyplot.show", side_effect=fake_show):
            return collect_points(Image.new("RGB", (2000, 1000)), names)

    def test_original_coordinates_undo_and_validation(self):
        result = self.run_picker([
            ("key", "enter"), ("click", (-100, 300)),
            ("click", (200, 300)), ("click", (900, 400)),
            ("key", "backspace"), ("click", (1000, 300)), ("click", (1500, 700)),
            ("right", (1500, 700)), ("click", (1600, 800)),
            ("click", (1700, 850)), ("key", "enter")])
        assert_allclose(list(result.values()), [[200, 300], [1000, 300], [1600, 800]], atol=1e-8)

    def test_close_and_escape_cancel(self):
        for action in (("close", None), ("key", "escape")):
            with self.subTest(action=action), self.assertRaises(SelectionCancelled):
                self.run_picker([("click", (200, 300)), action])


class CliTests(unittest.TestCase):
    def test_cli_saves_exact_size_and_metadata_without_gui(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.new("RGBA", (100, 100), (200, 0, 0, 128)).save(folder / "input.png")
            (folder / "points.json").write_text(json.dumps(
                {"image_left_eye": [25, 25], "image_right_eye": [75, 25]}))
            run = subprocess.run([sys.executable, str(ROOT / "main.py"), str(folder / "input.png"),
                "--points", str(folder / "points.json"), "--size", "301", "149",
                "--output", str(folder / "result.jpg"), "--metadata", str(folder / "result.json"),
                "--no-preview"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            with Image.open(folder / "result.jpg") as output:
                self.assertEqual(output.size, (301, 149))
                self.assertEqual(output.mode, "RGB")
            data = json.loads((folder / "result.json").read_text())
            self.assertEqual(data["output_size"], [301, 149])
            self.assertEqual(len(data["landmarks"]), 2)

    def test_refuse_overwriting_input_or_sharing_output_paths(self):
        for args in (["photo.png", "--output", "photo.png"],
                     ["photo.png", "--output", "out.png", "--metadata", "out.png"]):
            with self.subTest(args=args), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
