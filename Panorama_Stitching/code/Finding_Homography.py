import random
from datetime import datetime
import numpy as np

"""
Finding Homography : Then you should calculate a Homography Matrix for each pair of
sub-images (by using RANSAC method).

It was made efficient with the following method:
https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
"""
def calculate_homography_matrix(feature_1, feature_2, distance_threshold):

    num_of_points = len(feature_1)
    optimum_H = None
    max_inliers = 0
    i = 0
    while i < 100:
        random.seed(datetime.now())
        random_numbers = random.sample(range(0, num_of_points), 4)
        p_matrix = []
        for num in random_numbers:
            x = int(feature_1[num][0])
            y = int(feature_1[num][1])
            x_d = int(feature_2[num][0])
            y_d = int(feature_2[num][1])
            p_matrix.append([0, 0, 0, -x, -y, -1, x * y_d, y_d * y, y_d])
            p_matrix.append([x, y, 1, 0, 0, 0, -x * x_d, -x_d * y, -x_d])
        p_matrix.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        p_matrix = np.asarray(p_matrix).reshape(9, 9)

        if np.linalg.det(p_matrix) == 0:
            continue
        b_mat = []
        for ind in range(8):
            b_mat.append([0])
        b_mat.append([1])
        p_inv = np.linalg.inv(p_matrix)
        H_matrix = np.dot(p_inv, b_mat)
        H_matrix = np.reshape(H_matrix, (3, 3))

        distances = calc_distances(feature_1, feature_2, H_matrix)

        num_of_inliers = 0
        for distance in distances:
            if distance < distance_threshold:
                num_of_inliers += 1

        if num_of_inliers > max_inliers:
            max_inliers = num_of_inliers
            optimum_H = H_matrix
        i += 1

    x_crop = feature_1[0][0].astype(int)
    y_pos = feature_1[0][1].astype(int)

    prime_num = base_formula(x_crop, y_pos, optimum_H)
    x_d_crop = (prime_num[0] / prime_num[2])[0].astype(int)

    return x_crop, x_d_crop


def calc_distances(feature_1, feature_2, H):
    distances = []
    for i in range(len(feature_1)):
        x = feature_1[i][0].astype(int)
        y = feature_1[i][1].astype(int)

        x_y_prime_mat = base_formula(x, y, H)
        if x_y_prime_mat[2] == 0:
            x_y_prime_mat[2] = 0.001
        x_d = (x_y_prime_mat[0] / x_y_prime_mat[2])[0].astype(int)
        y_d = (x_y_prime_mat[1] / x_y_prime_mat[2])[0].astype(int)

        x_2 = feature_2[i][0].astype(int)
        y_2 = feature_2[i][1].astype(int)

        try:
            dist = np.math.sqrt((x_2 - x_d) ** 2 + (y_2 - y_d) ** 2)
            distances.append(dist)
        except:
            pass

    return distances


def base_formula(x, y, H):
    prime = [[x], [y], [1]]
    prime_inverse = np.dot(H, prime)
    return prime_inverse
