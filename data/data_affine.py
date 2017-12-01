from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import numpy as np
from skimage.transform import warp, AffineTransform


class RandomAffineTransform(object):
    def __init__(self,
                 scale_range=[0.4,1.05],
                 rotation_range=[-np.pi/8,np.pi/8],
                 shear_range=[-np.pi/32,np.pi/32],
                 translation_range=[-0.0, 0.0],
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img):
        img_data = np.array(img)
        h, w, n_chan = img_data.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)
        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
        img_data1 = warp(img_data, af.inverse)
        return img_data1
        img1 = Image.fromarray(np.uint8(img_data1 * 255))
        return img1
