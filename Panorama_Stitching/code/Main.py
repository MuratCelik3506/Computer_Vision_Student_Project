import glob
import os
import cv2 as cv
import numpy as np
import Feature_Extraction
import Feature_Matching
import Finding_Homography
import Merging_Transformation

"""
Murat Çelik
BBM418 - Assignment 2
Spring 2022

"""

def main(image_folder, result_dir, number ):
    all_images_list = glob.glob(image_folder)

    read_all_img_list = []
    for image_file_location in all_images_list:
        img = cv.imread(image_file_location, 0)
        read_all_img_list.append(img)
    img_list_2 = []
    image_name_index = 0
    time_counter = 0

    # HyperParameters
    desired_num_of_matches = 8
    distance_threshold = 50
    img_1_crop_ratio = 0.2
    img_2_crop_ratio = 0.2
    method = "orb" # change "sift"

    while len(read_all_img_list) > 1:
        img_index = 0
        while True:
            if img_index == len(read_all_img_list):
                break
            img_1 = read_all_img_list[img_index]

            if img_index + 1 == len(read_all_img_list):
                img_list_2.append(img_1)
                break
            img_2 = read_all_img_list[img_index + 1]

            # Feature Extraction
            keypoints_1, features_1, keypoints_2, features_2, time_counter = \
                Feature_Extraction.feature_extraction(method, img_1, img_2, time_counter)

            if features_1 is None or features_2 is None:
                #print("There is no match !")
                img_1 = img_1[:, 0:int(img_1.shape[1] * (1 - img_1_crop_ratio))]
                img_2 = img_2[:, int(img_2.shape[1] * img_2_crop_ratio):]
                merge_img = np.hstack((img_1, img_2))

                img_list_2.append(merge_img)
                img_index += 2
                continue

            #Feature Matching
            good_matches_list, good_matches = \
                Feature_Matching.find_matches(features_1, features_2, desired_num_of_matches, method)

            if len(good_matches) < desired_num_of_matches:
                #print("There is not enough match !")
                img_1 = img_1[:, 0:int(img_1.shape[1] * (1 - img_1_crop_ratio))]
                img_2 = img_2[:, int(img_2.shape[1] * img_2_crop_ratio):]
                merge_img = np.hstack((img_1, img_2))
                img_list_2.append(merge_img)
                img_index += 2
                continue

            feature_ps_img_1 = np.float32([keypoints_1[m[0].queryIdx].pt for m in good_matches_list])
            feature_ps_img_2 = np.float32([keypoints_2[m[0].trainIdx].pt for m in good_matches_list])

            # Finding Homograph
            x_crop, x_d_crop = \
                Finding_Homography.calculate_homography_matrix(feature_ps_img_1, feature_ps_img_2, distance_threshold)
            # Merging by Transformation
            merge_img = Merging_Transformation.merge_image(img_1, img_2, x_crop, x_d_crop)
            img_list_2.append(merge_img)

            img_index += 2
            image_name_index += 1

        read_all_img_list = img_list_2.copy()
        img_list_2.clear()
    print("time", time_counter)
    cv.imshow("im", read_all_img_list[0])
    panorama_im_path = result_dir + "\\" + method + str(number) +"_panorama.png"
    cv.imwrite(panorama_im_path, read_all_img_list[0])

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":

    for x in range(18,22):
        name = "part_1_dataset\cvc01passadis-cyl-pano"
        file_num = str(x)
        file_num = file_num.rjust(2, '0')
        image_folder = name + file_num + "\*g"
        try :
            os.mkdir(name + file_num + "_result")
        except FileExistsError:
            print("Already exist")
        result_dir = name + file_num + "_result"
        try:
            main(image_folder, result_dir,x)
        except:
            print("pass ",file_num )