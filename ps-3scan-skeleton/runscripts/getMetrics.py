import pickle

from skeleton.skeletonClass import Skeleton

"""
Program to run and save statistics after pruning
"""

path = input("enter root directory of the binary pngs")
skeleton = Skeleton(path)
skeleton.segmentStatsAfterPruning()
pickle.dump(skeleton.statsAfter, open("skeletonStas.p", "wb"))
