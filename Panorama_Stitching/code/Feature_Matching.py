import cv2 as cv

"""
Feature Matching : Then you are expected to code a matching function (for example this can
be based k-nearest neighbor method) to match extracted keypoints between pairs of sub-images. (For example,
Pair 1: (SubImage 1 - SubImage 2), Pair 2: (SubImage 2 - SubImage 3), Pair N: (SubImage N-1 - SubImage
N) ) 
"""
def create_bfmatcher(method):
    if method == 'sift' or method == 'surf':
        bf_matcher = cv.BFMatcher(cv.NORM_L2)
    elif method == 'orb':
        bf_matcher = cv.BFMatcher(cv.NORM_HAMMING)
    else:
        bf_matcher = cv.BFMatcher()
    return bf_matcher


def find_matches(feature_one, feature_two, minimum_match_num, method):
    bf = create_bfmatcher(method)
    matches_knn = bf.knnMatch(feature_one, feature_two, k=2)
    ratio = 0.2
    matches = list()
    while len(matches) < minimum_match_num and ratio < 0.8:
        ratio += 0.05
        matches = list()
        for m, n in matches_knn:
            if m.distance < ratio * n.distance:
                matches.append(m)
    matches = sorted(matches, key = lambda x: x.distance)
    matches_list = list()
    for match in matches:
        matches_list.append([match])
    return matches_list, matches
