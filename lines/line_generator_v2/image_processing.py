from config_parser import Config

import random
import os
import numpy as np
import matplotlib.pyplot as plt

from cv2 import getPerspectiveTransform, warpPerspective, imread, cvtColor, COLOR_RGB2GRAY, getStructuringElement, MORPH_ELLIPSE, imwrite, resize, GaussianBlur

from skimage.morphology import binary_dilation, thin

import pandas as pd

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
        self._image = np.ones((1024, 32)) * 255
        self._last_max_x = 32
        self._last_top_y = 128
        self._last_bottom_y = 1024 - 128
        self.metadata = []
        
    def _reset(self) -> None:
        self._image = np.ones((1024, 32)) * 255
        self._last_max_x = 32
        self._last_top_y = 128
        self._last_bottom_y = 1024 - 128
    
    def create_line(self, line: str) -> None:
        unicode_codes = ["u"+str(hex(ord(char)))[2:] for char in line.strip()]

        for i, u in enumerate(unicode_codes):
            self.add_character(u)
    
        self.write_current(line)
        self._reset()
    
    def write_current(self, line: str) -> None:
        filename = f"./generated/{len(self.metadata)}.png"        
        if self._image.shape[0] != 128:
            scale_factor = 128 / self._image.shape[0]
            self._image = resize(self._image, (int(self._image.shape[1] * scale_factor), 128))
        
        self._image = GaussianBlur(self._image, (5, 5), 0)
        mask1 = self._image >= 220
        mask0 = self._image < 220
        
        self._image[mask1] = 255
        self._image[mask0] = 0    
        
        imwrite(filename, self._image)
        self.metadata.append({"filename": filename, "line": line})
        
    def write_metadata(self, filename: str) -> None:
        df = pd.DataFrame(self.metadata)
        df.to_csv(filename, index=False)
        

    def _get_coords(self, unicode: str) -> np.array:
        try:
            char_height = self.config.get_height(unicode)
            char_width = self.config.get_width(unicode)
            char_rotation = self.config.get_rotation(unicode)
            char_translation_y = self.config.get_translation_x(unicode)
            char_translation_x = self.config.get_translation_y(unicode)
        except KeyError:
            raise KeyError(f"Missing character {unicode}")
        
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

        if new_max_x > self._image.shape[1]:
            self._image = np.pad(self._image, ((0, 0), (0, new_max_x - self._image.shape[1])), constant_values=255)

        self._last_max_x = new_max_x
        self._last_top_y = min(rotated_coords[:, 1])
        self._last_bottom_y = max(rotated_coords[:, 1])

        return rotated_coords.astype("float32")
    
    def _get_char(self, unicode: str) -> np.array:
        if unicode in ["u20"]:
            return np.ones((64,64)) * 255
        
        not_valid = True
        
        while not_valid:
            folder = f"../../cleaned/{unicode}/"
            char_path = folder + random.choice(os.listdir(folder))
            try:
                im = imread(char_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing character {unicode}")
            im = cvtColor(im, COLOR_RGB2GRAY)
            im = np.asarray(im)
            im = 255 - im
            if np.mean(im) / 255 > 0.5 and np.mean(im) / 255 < 0.85 or unicode in ["u2e"]:
                not_valid = False
            im[np.where(im == 0)] = 1

        return np.asarray(im)


    def _apply_transform(self, dst: np.array) -> None:
        """
        Add dst to self._image
        """
        assert dst.shape == self._image.shape

        mask = (dst == 1)

        self._image[mask] = dst[mask]
        self._image[self._image < 254] = 0


    def add_character(self, unicode: str) -> None:
        char_img = self._get_char(unicode)
        char_coords = np.asarray([[0, 0], [0, char_img.shape[0]], [char_img.shape[1], 0], [char_img.shape[1], char_img.shape[0]]]).astype("float32")
        map_coords = self._get_coords(unicode)

        M = getPerspectiveTransform(char_coords, map_coords)

        dst = warpPerspective(char_img, M, (self._image.shape[1], self._image.shape[0]))

        self._apply_transform(dst)


if __name__ == "__main__":
    transformer = ImageTransformer("config.yml")
    print(transformer.metadata)

    code_file = "apl.txt"
    with open(code_file, "r") as f:
        lines = f.readlines()
        
    for i, random_line in enumerate(lines):
        if i % 100 == 0:
            print(f"Processed {i} lines out of 10000")
        try:
            # random_line = random.choice(lines)
            if len(random_line) > 50:
                continue
            
            # lines.remove(random_line)
                
            transformer.create_line(random_line[:-1])
        except (FileNotFoundError, KeyError) as e:
            print(e)
            transformer._reset()
        
    transformer.write_metadata("metadata.csv")
