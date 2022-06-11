import numpy as np


"""
Merging by Transformation : Merge sub-images into single panorama by applying transfor-
mation operations on sub-images by using the Homography Matrix. 
"""
def merge_image(img_one, img_two, x_crop, x_d_crop):

    img_one = img_one[:, 0:x_crop]
    img_two = img_two[:, x_d_crop:]

    merge_img = np.hstack((img_one, img_two))
    return merge_img
