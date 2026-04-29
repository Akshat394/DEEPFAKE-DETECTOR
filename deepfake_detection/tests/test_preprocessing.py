import numpy as np

from src.preprocessing.ascii_conversion import ASCIIConverter


def test_ascii_converter_shape():
    conv = ASCIIConverter(grid_size=(80, 40), ascii_chars=".+=@*%#")
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    ascii_img = conv.convert_frame(frame)
    assert ascii_img.shape == (80, 40, 3)
