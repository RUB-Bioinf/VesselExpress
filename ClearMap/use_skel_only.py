import time
import ClearMap.IO.IO as io
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl
import numpy as np

start_time = time.time()

path = "/bph/puredata1/bioinfdata/user/phispa/Test/"
image = "li_Frangi_B11_ipsi_comp"

seg = io.read(path + image + ".tif")
seg = seg * 1
np.save(path + image + ".npy", seg)

print("Skeletonization")
sink_skel = path + "skel_" + image
im_skel = skl.skeletonize(path + image + '.npy', sink=sink_skel + '.npy', delete_border=True, verbose=True)

print("Write skeletonization result to tif")
im_skel.array = im_skel.array * 255
io.write(sink_skel + ".tif", im_skel)

