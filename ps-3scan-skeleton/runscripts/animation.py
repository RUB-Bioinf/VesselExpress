import os

import numpy as np

from mayavi.api import Engine
from mayavi import mlab

"""
Create an animation using mayavi, will create a series of images.
use KESMAnalysis.cli.makemp4.py to create a video or
use imagemagick to create video from this frames
convert -set delay 20 -loop 0 -quality 1000 -scale 100% *.png /home/pranathi/animExp.mpg
"""
BOUNDINGBOXCOLOR = (0, 0, 0)  # Color of bounding box around the volume
ELEVATIONANGLE = 102.041  # just above 90 to display 3D nature of the images
POINTSSIZE = 6.448  # Size of sphere isosurface around points on skeleton
THRESHOLDCONTOURLIST = [0.7525]  # Contour list of threshold volume, closer to white (in BW grey colormap)
THRESHOLDSCENEOPACITY = 0.3025  # Make threshold volume closer to transparent (0=fully transparent)
SKELETONCONTOURLIST = [0.9901]  # Higher the contours, closer to red (in BGR colormap)
WHITEBACKGROUND = (1, 1, 1)  # Scene background is set to white


def getFrames(pathThresh, pathSkel, totalTime, fps=24, totalRotation=360):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    pathThresh : str
        path of the .npy thresholded 3D Volume

    pathSKel : str
        path of the .npy skeleton 3D Volume

    totalTime : integer
        in seconds, duration of the video

    fps : integer
        frames per second, number of input frames per second

    totalRotation : integer
        angle in degrees frames should be captured in, integer between 0 and 360

    Returns
    -------
    frames of png images in the same directory as pathThresh
        mayavi scenes are saved as png images at different
        angle of rotations as anim%i.png i is the ith frame

    Notes
    -----
    threshold and skeletonized volume are overlapped,
    they need not be of the same size, but assuming they are
    for the same volume it asserted that they are of same size
    thresholded volume's isosurface is transparent and
    in grey and the skeletonized volume can be seen through
    it and is in red
    totalRotation can be any angle but it will be adjusted between 0 and
    360 using the % (modulus) operator
    """
    # modulus of totalRotation
    totalRotation = totalRotation % 360
    # total frames
    totalFrameCount = fps * totalTime
    # degree of rotation after each frame
    degreePerFrame = totalRotation / totalFrameCount
    # load the threshold and skeleton paths
    threshold = np.load(pathThresh)
    skeleton = np.load(pathSkel)
    assertionStr = "threshold and skeleton  must be of same shape"
    assert threshold.shape == skeleton.shape, (assertionStr, threshold.shape, skeleton.shape)
    mayaviEngine = Engine()
    mayaviEngine.start()
    # Create a new mayavi scene.
    mayaviScene = mayaviEngine.new_scene()
    mayaviScene.scene.background = WHITEBACKGROUND
    # thresholded image in transparent grey
    thresholdScene = mlab.contour3d(np.uint8(threshold), colormap='gray', contours=THRESHOLDCONTOURLIST)
    thresholdScene.actor.property.opacity = THRESHOLDSCENEOPACITY
    # skeleton in red
    f = mlab.contour3d(np.uint8(skeleton), contours=SKELETONCONTOURLIST)
    f.actor.property.representation = 'points'
    f.actor.property.point_size = POINTSSIZE
    mlab.options.offscreen = True
    mlab.outline(f).actor.property.color = BOUNDINGBOXCOLOR
    # extract rootDir of pathThresh
    rootDir = os.path.split(pathThresh)[0] + os.sep
    # Make an animation:
    for i in range(totalFrameCount):
        # Rotate the camera by 10 degrees.
        mayaviScene.scene.camera.azimuth(degreePerFrame)
        mayaviScene.scene.camera.elevation(ELEVATIONANGLE)
        # Resets the camera clipping plane so everything fits and then
        # renders.
        mayaviScene.scene.reset_zoom()
        # Save the scene. magnification=4 gives saves as an image when seen in fullscreen
        mayaviScene.scene.magnification = 4
        mayaviScene.scene.save(rootDir + "anim%d.png" % i)
