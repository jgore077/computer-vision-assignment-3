"""
We will use ORB as the detection algorithm
https://docs.opencv2.org/4.x/db/d95/classcv2_1_1ORB.html
https://docs.opencv2.org/4.x/d1/d89/tutorial_py_orb.html

For feature matchers we have to be aware of certain parameters
"""
from matplotlib import pyplot as plt
import cv2
from funcs import *
import numpy as np


pair_dict={
    "./pairs/Boston.jpeg":"./pairs/Boston1.jpeg",
    "./pairs/Castle.jpg":"./pairs/Castle1.jpg",
    "./pairs/MountRushmore.jpg":"./pairs/MountRushmore1.jpg",
    "./pairs/NotreDame.jpg":"./pairs/NotreDame1.jpg"
}
 
n_features=1000
edge_threshold=10
scale_factor=2
SIZE=600

# Initiate ORB detector
orb = cv2.ORB_create(n_features,scale_factor,edgeThreshold=edge_threshold)
# Initialize brute force matcher
orb_bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

# Create sift detector with your initial params
sift=cv2.SIFT_create(100,4,0.06,15,1.9)
# Create sift matcher
sift_bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)

iterate_image_pairs("ORB and BFMatcher",pair_dict,SIZE,"ORB",orb,orb_bf)
iterate_image_pairs("SIFT and BFMatcher",pair_dict,SIZE,"SIFT",sift,sift_bf)  
iterate_image_pairs("SIFT and FLANN",pair_dict,SIZE,"SIFT",sift,matcher_type="FLANN")  