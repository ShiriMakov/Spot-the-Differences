
import numpy as np
import cv2
import matplotlib.pyplot as plt


# #####################################################################################################################


def covert_to_grayscale(im1, im2):
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    return im1_gray, im2_gray


# #####################################################################################################################

def orb_rigid_alignment(im1, im2, max_features, good_match_percent, diffy_thresh, data_path):
    im1_gray, im2_gray = covert_to_grayscale(im1, im2)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    sorted_matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    num_good_matches = int(len(sorted_matches) * good_match_percent)
    good_matches = sorted_matches[:num_good_matches]

    # Extract location of good matches and filter by diffy if rotation is small
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # initialize empty arrays for newpoints1 and newpoints2 and mask
    newpoints1 = np.empty(shape=[0, 2], dtype=np.float32)
    newpoints2 = np.empty(shape=[0, 2], dtype=np.float32)
    matches_Mask = [0] * len(good_matches)

    count = 0
    for i in range(len(good_matches)):
        pt1 = points1[i]
        pt2 = points2[i]
        pt1x, pt1y = zip(*[pt1])
        pt2x, pt2y = zip(*[pt2])
        diffy = np.float32(np.float32(pt2y) - np.float32(pt1y))
        if abs(diffy) < diffy_thresh:
            newpoints1 = np.append(newpoints1, [pt1], axis=0).astype(np.uint8)
            newpoints2 = np.append(newpoints2, [pt2], axis=0).astype(np.uint8)
            matches_Mask[i] = 1
            count += 1

    # Find Affine Transformation
    # note swap of order of newpoints here so that image2 is warped to match image1
    m, inliers = cv2.estimateAffinePartial2D(newpoints2, newpoints1)

    # Use affine transform to warp im2 to match im1
    height, width, channels = im1.shape
    image2Reg = cv2.warpAffine(im2, m, (width, height))

    return image2Reg, m


# ########################################################################################################################

# ECC (Enhanced Correlation Coefficient Maximization): rigid warping
def ECC_rigid_alignment(im1, im2, number_of_iterations, termination_eps, data_path, h_matrix=[0]):
    im1_gray, im2_gray = covert_to_grayscale(im1, im2)

    # Find size of image1
    sz = im1.shape

    # Define the motion model - euclidean is rigid (SRT)
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 matrix and initialize the matrix to identity matrix I (eye)
    if len(h_matrix) > 1:
        warp_matrix = h_matrix  # init with the warping mat calculated on the low-resolution images
    else:
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)

    # Warp im2 using affine
    if warp_mode == cv2.MOTION_HOMOGRAPHY:  # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:  # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2_aligned, warp_matrix


# #####################################################################################################################
# thresholding
def Otsus_thresholding(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, new_inage = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return new_inage

# #####################################################################################################################

# ground_truth solution
def show_ground_truth(im, case_number):
    # mark defects from the ground-truth results
    ground_truth_map = im.copy()
    ground_truth_map[:] = 0

    if case_number == 1:
        ground_truth_map[146:152, 331:337] = 255
        ground_truth_map[79:85, 242:248] = 255
        ground_truth_map[94:100, 79:85] = 255

    elif case_number == 2:
        ground_truth_map[340:344, 260:265] = 255
        ground_truth_map[100:105, 103:108] = 255
        ground_truth_map[75:80, 257:262] = 255

    ground_truth_map = np.fliplr(ground_truth_map)
    ground_truth_map = np.rot90(ground_truth_map)

    return ground_truth_map
