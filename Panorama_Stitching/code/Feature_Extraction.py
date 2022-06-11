import time
import cv2 as cv
"""
Feature Extraction (10 Points): Firstly you are expected to the extract keypoints in the sub-images by a
keypoint extraction method (SIFT/SURF and ORB)
"""
def feature_extraction(method, img_one, img_two, time_counter ):
    if method == "sift":
        kp_desc_method = cv.SIFT_create()
    elif method == "surf":
        kp_desc_method = cv.SURF_create()
    elif method == "orb":
        kp_desc_method = cv.ORB_create()

    start_method = time.time()
    keypoints_1, features_1 = kp_desc_method.detectAndCompute(img_one, None)
    keypoints_2, features_2 = kp_desc_method.detectAndCompute(img_two, None)
    final_time_method = (time.time() - start_method)
    time_counter += final_time_method
    return keypoints_1, features_1, keypoints_2, features_2, time_counter
