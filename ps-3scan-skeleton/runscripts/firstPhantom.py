import math
import random

from skimage.draw import ellipse
import numpy as np
import scipy


def makeFakeVessels(imgsize=(512, 512), background=230):
    """
    create and save a matrix with whitish background and randomly selected vessel sizes and save matrices generated as images of format png
    """
    nVes = 20
    mu = 20
    sigma = 5
    minw = 5
    sx, sy = imgsize
    vasc = np.ones((sx, sy), dtype=np.uint8) * background

    for i in range(nVes):
        cx, cy = random.uniform(0, sx), random.uniform(0, sy)
        r1, r2 = 0, 0
        while (r1 < minw) or (r2 < minw):
            np.random.seed(20)
            r1 = np.random.normal(mu, sigma)
            r2 = np.random.normal(mu, sigma)
        print(r1, r2)

        rr, cc = ellipse(cy, cx, r1, r2)
        if np.any(rr >= sy):
            ix = rr < sy
            rr, cc = rr[ix], cc[ix]
        if np.any(cc >= sx):
            ix = cc < sx
            rr, cc = rr[ix], cc[ix]
        vasc[rr, cc] = 1  # make circle blackish
    return vasc


def eulerAnglesToRotationMatrix(theta):
    # Calculates Rotation Matrix given euler angles.
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    return np.array((R_x, R_y, R_z))


def getPhantom(slices):
    vessels = _getCrosssection()
    phantom = np.zeros((slices, vessels.shape[0], vessels.shape[1]), dtype=np.uint8)
    for i in range(0, slices):
        phantom[i, :, :] = vessels
    return phantom


def _getCrosssection():
    s = np.array((512, 512))
    vessels = makeFakeVessels(s, background=0)
    return vessels


def getPhantomLineToCheckOrientation(size=(25, 25, 25)):

    h_line = np.zeros(size, dtype=bool)
    h_line[3, :, 4] = 1
    v_lines = h_line.T.copy()

    # A "comb" of lines
    h_lines = np.zeros(size, dtype=bool)
    h_lines[0, ::3, :] = 1
    v_liness = h_lines.T.copy()
    # A grid made up of two perpendicular combs
    grid = h_lines | v_liness
    stationaryImages = [h_line, v_lines, h_lines, v_liness, grid]
    return stationaryImages


if __name__ == '__main__':
    phantom = getPhantom(424)
    np.save('/home/pranathi/Downloads/phantom.npy', phantom)
    cylinderRotated = scipy.ndimage.interpolation.rotate(phantom, 45)
