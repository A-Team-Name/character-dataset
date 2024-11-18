import os
import random

import numpy as np
from PIL import Image
from config_parser import Config


class ImageTransformer:
    @staticmethod
    def _get_first_index(arr: list[int], item: int):
        for i, val in enumerate(arr):
            if val == item:
                return i
            
    def __init__(self, config_file: str):
        self.config = Config(config_file)

    def _rotate_image(self, image: np.ndarray, degrees: int) -> np.ndarray:
        pil_image = Image.fromarray(image)
        rgba_img = pil_image.convert("RGBA")
        rot_img = rgba_img.rotate(degrees, expand=1)
        fff = Image.new("RGBA", rot_img.size, (255,)*4)
        out = Image.composite(rot_img, fff, rot_img).convert(pil_image.mode)
        return np.asarray(out)

    def rotate_image(self, image: np.ndarray, unicode: str) -> np.ndarray:
        degrees = self.config.get_rotation(unicode)

        return self._rotate_image(image, degrees)

    def _pad_image(self, image: np.ndarray, pad_pixels: int) -> np.ndarray:
        if pad_pixels > 0:
            return np.insert(image, 0, [[255]]*pad_pixels, axis=0)
        return image

    def pad_image(self, image: np.ndarray, unicode: str) -> np.ndarray:
        pad_amount = self.config.get_padding(unicode)

        return self._pad_image(image, pad_amount)

    def _translate_image(self, image: np.ndarray, translate_pixels: int) -> np.ndarray:  
        if translate_pixels < 0:
            translate_pixels *= -1
            return np.insert(image, 0, [[255]]*translate_pixels, axis=1)
        
        if translate_pixels > 0:
            return np.insert(image, image.shape[1], [[255]]*translate_pixels, axis=1)
        
        return image

    def translate_image(self, image: np.ndarray, unicode: str) -> np.ndarray:
        translate_amount = self.config.get_translation(unicode)

        return self._translate_image(image, translate_amount)

    def convert_height(self, image: np.ndarray, new_height: int) -> np.ndarray:
        # TODO: FIX!!!
        return image

    def _stitch_image(self, left: np.ndarray, right: np.ndarray, stitch_type: str) -> np.ndarray:
        if stitch_type == "BOTTOM":
            if left.shape[0] > right.shape[0]:
                right = np.insert(right, 0, [[255]]*(left.shape[0]-right.shape[0]), axis=0)
            elif left.shape[0] < right.shape[0]:
                left = np.insert(left, 0, [[255]]*(right.shape[0]-left.shape[0]), axis=0)
        elif stitch_type == "TOP":
            if left.shape[0] > right.shape[0]:
                right = np.insert(right, right.shape[0], [[255]]*(left.shape[0]-right.shape[0]), axis=0)
            elif left.shape[0] < right.shape[0]:
                left = np.insert(left, left.shape[0], [[255]]*(right.shape[0]-left.shape[0]), axis=0)
        elif stitch_type == "MIDDLE":
            if left.shape[0] > right.shape[0]:
                right = np.insert(right, right.shape[0], [[255]]*((left.shape[0]-right.shape[0])//2), axis=0)
                right = np.insert(right, 0, [[255]]*(left.shape[0]-right.shape[0]), axis=0)
            elif left.shape[0] < right.shape[0]:
                left = np.insert(left, left.shape[0], [[255]]*((right.shape[0]-left.shape[0])//2), axis=0)
                left = np.insert(left, 0, [[255]]*(right.shape[0]-left.shape[0]), axis=0)

        print(left.shape, right.shape)

        return np.concatenate((left, right), axis=1)
    
    def stitch_image(self, im: np.ndarray, new_char: np.ndarray, unicode: str) -> np.ndarray:
        stitch_type = self.config.get_stitch(unicode)

        return self._stitch_image(im, new_char, stitch_type)

    def get_character(self, unicode: str) -> np.ndarray:
        chars = os.listdir(f"../processed/{unicode}")

        random_char = random.choice(chars)

        img = Image.open(f"../processed/{unicode}/{random_char}")

        return np.asarray(img)

    def _bounding_box(self, image: np.ndarray) -> np.ndarray:
        # function to get the character into a bounding box
        valid_rows = np.min(image, axis=0)
        valid_cols = np.min(image, axis=1)
        top = max(0, ImageTransformer._get_first_index(valid_rows, 0)-10)
        bottom = min(len(valid_rows) - ImageTransformer._get_first_index(reversed(valid_rows), 0) + 10, len(valid_rows))

        left = max(0, ImageTransformer._get_first_index(valid_cols, 0)-10)
        right = min(len(valid_cols) - ImageTransformer._get_first_index(reversed(valid_cols), 0) + 10, len(valid_cols))

        return image[left:right, top:bottom]

    def add_character(self, im: np.ndarray, new_char: str) -> np.ndarray:
        if new_char == "u20":
            temp_img = np.asarray(Image.new(mode="L", size=(64, 64), color=255))
        else:
            char_img = self.get_character(new_char)

            temp_img = self.rotate_image(char_img, new_char)
            temp_img = self._bounding_box(temp_img)
            temp_img = self.pad_image(temp_img, new_char)
            temp_img = self.translate_image(temp_img, new_char)
            temp_img = self.convert_height(temp_img, new_char)

        if im is None:
            return temp_img

        return self.stitch_image(im, temp_img, new_char)


if __name__ == "__main__":
    img_transformer = ImageTransformer("example_individual_config.yml")

    code_file = "python.txt"
    random_line = ""
    with open(code_file, "r") as f:
        random_line = random.choice(f.readlines())
        
    print(random_line)
    unicode_codes = ["u"+str(hex(ord(char)))[2:] for char in random_line.strip()]

    im = None
    for unicode in unicode_codes:
        im = img_transformer.add_character(im, unicode)

    Image.fromarray(im).save("temp2.png")