import numpy as np
from PIL import Image
from opensoundscape.preprocess.img_augment import time_split


def test_basic_split():
    shape = (1, 5, 3)
    arr = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=np.uint8
    ).reshape(shape)
    img = Image.fromarray(arr)
    img = time_split(img, seed=1)
    new_arr = np.array(img)
    assert arr.shape == shape
    assert new_arr.shape == shape
    assert np.all(arr != new_arr)
