from math import pi, cos, sin
from pylab import mgrid
from scipy import misc, spatial
import itertools
import os
import numpy as np
import random
import time


# Vertical Cylinder
def cylinder(size, xi, yi, r):
    x, y, z = mgrid[-1.0:1.0:1j * size[0],
                    -1.0:1.0:1j * size[1],
                    -1.0:1.0:1j * size[2]]

    return 1 * (np.sqrt((x - xi)**2 + (y - yi)**2) < r)


def makeThreebars():
    c = cylinder((50,50,50), 0.5, 0.5, 1.0 / 6)
    n = c.copy()

    n = np.logical_or(n, np.transpose(c, [2,1,0])[:,::-1,:])
    n = np.logical_or(n, np.transpose(c, [0,2,1])[::-1,:,::-1])
    return n


def makePoint(size, xi, yi, zi, r):
    x, y, z = mgrid[-1.0:1.0:1j * size[0],
                    -1.0:1.0:1j * size[1],
                    -1.0:1.0:1j * size[2]]

    return 1 * (np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) < r)


def traceFunction(xf, yf, zf, tBracket, nSteps, radaii, shape=(100,100,100)):
    # Compute the time offsets
    tMin, tMax = tBracket
    ts = np.linspace(tMin, tMax, nSteps)

    # Compute the x,y,z loci
    xs = xf(ts)
    ys = yf(ts)
    zs = zf(ts)

    x, y, z = mgrid[-1.0:1.0:1j * shape[0],
                    -1.0:1.0:1j * shape[1],
                    -1.0:1.0:1j * shape[2]]

    s = np.zeros(shape, dtype=np.uint8)
    # Compute a sphere at each point and
    for xp, yp, zp in zip(xs, ys, zs):
        s = np.logical_or(s, makePoint(shape, xp, yp, zp, radaii))

    return s


def xyAnnulus(r, R):
    xf = lambda t: cos(t) * R
    yf = lambda t: sin(t) * R
    zf = lambda t: sin(4 * t) * 0

    c = traceFunction(xf, yf, zf, [0,2 * pi], nSteps=200, radaii=r)

    return c


def roseCurve(k):
    R = 0.9
    xf = lambda t: cos(k * t) * cos(t) * R
    yf = lambda t: cos(k * t) * sin(t) * R
    zf = lambda t: 0 * t

    c = traceFunction(xf, yf, zf, [0,k * 2 * pi], nSteps=k * 200, radaii=0.05)

    return c


def doLineSegment(xyz1, xyz2, shape=(100,100,100), spacing=1.0 / 50):
    # Break into xyzs
    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2

    length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    nPts = np.ceil(length / spacing)

    xs = np.linspace(x1, x2, nPts)
    ys = np.linspace(y1, y2, nPts)
    zs = np.linspace(z1, z2, nPts)

    s = np.zeros(shape, dtype=np.uint8)
    # Compute a sphere at each point and
    for xp, yp, zp in zip(xs, ys, zs):
        s = np.logical_or(s, makePoint((100,100,100), xp, yp, zp, spacing * 2))

    return s

# save("wiggle.npy", xyAnnulus(0.05, 0.5) )
# save("xyzBars.npy", makeThreebars())
# for x in range(1, 5):
#     save("roseCurve%i.npy" % x, roseCurve(x))


def voronoiVessels(nPoints=10):
    # Generate a bunch of points that pass the box extents slightly
    xyz = random.uniform(-1.2, 1.2, size=(nPoints, 3))

    # Triangulate them
    dt = spatial.Delaunay(xyz)

    # Set of already traced edges (2x speedup)
    tracedEdges = set()

    solid = np.zeros((100,100,100), dtype=np.uint8)
    # Go throrough the tetrahedra
    for n, tet in enumerate(dt.vertices):
        print(n, dt.vertices.shape)
        for eTuple in itertools.permutations(tet, 2):
            # Sort the edge indices to make matches
            eList = list(eTuple)
            eList.sort()
            eRef = "%i-%i" % tuple(eList)

            # If we already traced this edge, skip it
            if eRef in tracedEdges:
                continue
            tracedEdges.add(eRef)

            # Dereference the points of interest
            idx1, idx2 = eList
            pt1 = xyz[idx1,:]
            pt2 = xyz[idx2,:]

            # Trace the line onto our list
            solid |= doLineSegment(pt1, pt2)

    return solid


def makeImageStack(datacube, outputFolder):

    # Make the directory if necessary
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Concoct a filename
    x, y, z = 145.3573, 29.8706, 3.5409
    t = 0.001000
    v = 19.999

    # Cast to uint8 to avoid image bs
    newCube = np.zeros(datacube.shape, dtype=np.uint8)
    newCube[datacube >= 1] = 255

    for planarSlice in newCube:
        # Compute a filename
        timeString = time.strftime("%Y%m%d_%H%M%S")
        offsetsString = "x%05fy%05fz%05f" % (x, y, z)
        filename = "_".join([timeString, offsetsString, "t%05f" % t, "v%05f" % v]) + ".jpg"

        # Compute the full path
        filePath = os.path.join(outputFolder, filename)

        # Save the image
        misc.imsave(filePath, planarSlice)

        z += t


def annularAbuse1(r, R):
    xf = lambda t: cos(t) * R
    yf = lambda t: (sin(t) * R) + R
    zf = lambda t: sin(4 * t) * 0

    c = traceFunction(xf, yf, zf, [0,2 * pi], nSteps=200, radaii=r, shape=(100,100,100))
    xf = lambda t: cos(t) * R
    yf = lambda t: (sin(t) * R) - R
    zf = lambda t: sin(4 * t) * 0

    c += traceFunction(xf, yf, zf, [0,2 * pi], nSteps=200, radaii=r, shape=(100,100,100))

    return c


def annularAbuse2(r, R):
    xf = lambda t: cos(t) * R
    yf = lambda t: (sin(t) * R) + R
    zf = lambda t: sin(4 * t) * 0

    c = traceFunction(xf, yf, zf, [0,2 * pi], nSteps=200, radaii=r)

    zf = lambda t: cos(t) * R
    yf = lambda t: (sin(t) * R) - R
    xf = lambda t: sin(4 * t) * 0

    c += traceFunction(xf, yf, zf, [0,2 * pi], nSteps=200, radaii=r)

    return c


# makeImageStack(makeThreebars(), "threeBars")
# makeImageStack(xyAnnulus(.1,.9), "annulus")
# makeImageStack(annularAbuse1(.1,1), "xPlanar")
# makeImageStack(annularAbuse2(.1,1), "xTwist")
# makeImageStack(annularAbuse1(.1,.9), "largeXPlanar")
# phantoms = os.listdir("/home/pranathi/Desktop/phantoms/")

# skeleton_cythons = ['skeleton_cython_single_inclined_cylinder_greater_radius.npy',
#  'skeleton_cython_noninclined_many_cylinders.npy',
#  'skeleton_cython_tree.npy',
#  'skeleton_cython_single_inclined_cylinder_lesser_radius_diff_inclination.npy',
#  'skeleton_cython_single_inclined_cylinder_lesser_radius.npy',
#  'skeleton_cython_single_inclined_cylinder_radius=35_middle.npy',
#  'skeleton_cython_single_inclined_cylinder_lesser_radius=20.npy',
#  'skeleton_cython_single_inclined_cylinder_radius=35_first_to_last.npy']
# for ph, cys in zip(phantoms, skeleton_cythons):
#     ph = np.amax(np.load("/home/pranathi/Desktop/phantoms/" + ph), 0)
#     cys = np.amax(np.load("/home/pranathi/Desktop/skeleton_cythons/" + cys), 0)
#     plt.subplot(2, 1, 1)
#     plt.imshow(np.uint8(ph), cmap='gray')
#     plt.subplot(2, 1, 2)
#     plt.imshow(cys, cmap='gray')
#     plt.show()
