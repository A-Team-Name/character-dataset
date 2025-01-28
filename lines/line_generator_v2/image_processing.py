from config_parser import Config

import random
import os
import numpy as np
import matplotlib.pyplot as plt

from cv2 import getPerspectiveTransform, warpPerspective, imread, cvtColor, COLOR_RGB2GRAY, getStructuringElement, MORPH_ELLIPSE, imwrite

from skimage.morphology import binary_dilation, thin

def calculate_rotation(coords: np.array, rotation: int) -> np.array:
    """
    coords: (4, 2) np array consisting of 4 corners of rectangle to be rotated
    rotation: degrees
    """
    radians = -np.pi * (rotation / 180)
    center = np.sum(coords, axis=0)/4
    
    coords -= center

    rot = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])

    applied_rotation = (np.matmul(rot, coords.T)).T

    applied_rotation += center

    return applied_rotation

class ImageTransformer:
    def __init__(self, config_file: str):
        self.config = Config(config_file)
        self._image = np.zeros((1024, 32))
        self._last_max_x = 32
        self._last_top_y = 128
        self._last_bottom_y = 1024 - 128

    def show_current(self):
        print(self._image.shape)
        imwrite("temp3.png", self._image)
        plt.imshow(self._image, cmap="gray")
        plt.axis("off")
        plt.savefig("temp.png")

    def _get_coords(self, unicode: str) -> np.array:
        char_height = self.config.get_height(unicode)
        print(char_height)
        char_width = self.config.get_width(unicode)
        char_rotation = self.config.get_rotation(unicode)
        char_translation_y = self.config.get_translation_x(unicode)
        char_translation_x = self.config.get_translation_y(unicode)
        print("width", char_width)
        print("height", char_height)
        print("rotation", char_rotation)
        print("translation_x", char_translation_x)
        print("translation_y", char_translation_y)
        resulting_coords = np.array(
            [
                [self._last_max_x, 32],
                [self._last_max_x, 32+char_height],
                [self._last_max_x+char_width, 32],
                [self._last_max_x+char_width, 32+char_height]
                ]
            ).astype("float")
        
        rotated_coords = calculate_rotation(resulting_coords, char_rotation).astype("int")

        rotated_coords[:, 1] += char_translation_x
        rotated_coords[:, 0] += char_translation_y

        new_max_x = max(rotated_coords[:, 0])

        self._image = np.pad(self._image, ((0, 0), (0, new_max_x - self._image.shape[1])), constant_values=0)

        self._last_max_x = new_max_x
        self._last_top_y = min(rotated_coords[:, 1])
        self._last_bottom_y = max(rotated_coords[:, 1])

        return rotated_coords.astype("float32")
    
    def _get_char(self, unicode: str) -> np.array:
        if unicode in ["u20"]:
            return np.zeros((64,64))
        
        not_valid = True
        
        while not_valid:
            folder = f"../../cleaned/{unicode}/"
            char_path = folder + random.choice(os.listdir(folder))
            im = imread(char_path)
            im = cvtColor(im, COLOR_RGB2GRAY)
            im = np.asarray(im)
            if np.mean(im) / 255 > 0.2 and np.mean(im) / 255 < 0.4 or unicode in ["u2e"]:
                not_valid = False
            im[np.where(im == 255)] = 254

        return np.asarray(im)


    def _apply_transform(self, dst: np.array) -> None:
        """
        Add dst to self._image
        """
        assert dst.shape == self._image.shape

        mask = (dst == 254)

        self._image[mask] = dst[mask]
        self._image[self._image > 1] = 255


    def add_character(self, unicode: str) -> None:
        char_img = self._get_char(unicode)
        char_coords = np.asarray([[0, 0], [0, char_img.shape[0]], [char_img.shape[1], 0], [char_img.shape[1], char_img.shape[0]]]).astype("float32")
        map_coords = self._get_coords(unicode)
        print(char_coords)
        print(map_coords)

        M = getPerspectiveTransform(char_coords, map_coords)

        dst = warpPerspective(char_img, M, (self._image.shape[1], self._image.shape[0]))

        self._apply_transform(dst)


if __name__ == "__main__":
    transformer = ImageTransformer("config.yml")

    code_file = "../python.txt"
    random_line = """messages[-1]["content"] += "\\n" + p["content"]"""
    with open(code_file, "r") as f:
        random_line = random.choice(f.readlines())
        
    print(random_line)
    unicode_codes = ["u"+str(hex(ord(char)))[2:] for char in random_line.strip()]

    for i, u in enumerate(unicode_codes):
        print(u)
        transformer.add_character(u)
        
    transformer.show_current()