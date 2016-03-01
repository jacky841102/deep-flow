import numpy
from PIL import Image

from deep_matching.deep_matching_gpu import deepmatching
from deepflow2 import deepflow2

im1 = numpy.array(Image.open('sintel1.png'))
im2 = numpy.array(Image.open('sintel2.png'))
matches = deepmatching(im1, im2)
flow = deepflow2(im1, im2, matches, '-sintel')