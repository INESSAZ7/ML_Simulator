import utils
from skimage import data


def test_rotated_image():
    """test for rotate image"""
    image = data.cat()
    result = utils.rotated_image(image)
    assert image.shape == result.shape
