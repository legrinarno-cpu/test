from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class FaceExtractor:
    def __init__(self, size=(100, 100)):
        self.size = size

    def get_eyes(self, img):
        # Input: image. Output: left and right eye coordinates.
        plt.imshow(img)
        plt.axis("off")

        eyes = []

        for name in ("left", "right"):
            plt.title(f"Click eye on the {name}")
            point = plt.ginput(1, timeout=-1)[0]
            eyes.append(point)

            plt.plot(*point, "gx")
            plt.draw()

        plt.close()
        return eyes

    def extract(self, img):
        left, right = self.get_eyes(img)

        dx = right[0] - left[0]
        dy = right[1] - left[1]

        angle = np.degrees(np.arctan2(dy, dx))
        rotated = img.rotate(angle, expand=False)

        w, h = img.size
        centre = np.array([w / 2, h / 2])

        theta = np.radians(-angle)

        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Update eye positions after rotation.
        left = rotation @ (np.array(left) - centre) + centre
        right = rotation @ (np.array(right) - centre) + centre

        scale = np.linalg.norm(right - left)

        x, y = left

        # Crop relative to the aligned eye positions.
        box = (
            int(x - scale / 2),
            int(y - scale / 2),
            int(x + 1.5 * scale),
            int(y + 1.5 * scale)
        )

        return rotated.crop(box).resize(self.size)

def main():
    img = Image.open("1.jpg")

    extractor = FaceExtractor()
    face = extractor.extract(img)

    plt.imshow(face)
    plt.axis("off")
    plt.title("Extracted Face")
    plt.show()

if __name__ == "__main__":
    main()
