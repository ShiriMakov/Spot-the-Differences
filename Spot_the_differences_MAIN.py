
import os
import matplotlib.pyplot as plt
import cv2
import functions as f


# Load images
data_path = input('Please enter the path where your images are stored: ')
inspected_image = cv2.imread(os.path.join(data_path, "inspected_image.jpg"))
reference_image = cv2.imread(os.path.join(data_path, "reference_image.jpg"))

# inits ###################################################################################################
# ECC inits
number_of_iterations = 5000
termination_eps = 1e-4  # threshold of the increment in the correlation coefficient between two iterations
# ORB inits
max_features = 5000
good_match_percent = 0.5
diffy_thresh = 20

# ###################################################################################################

# perform ECC_rigid_alignment to get the right form of transformation matrix
temp_im, h_matrix1 = f.ECC_rigid_alignment(inspected_image, reference_image,
                                                       number_of_iterations,
                                                       termination_eps, data_path)

# align using ORB
temp_im, h_matrix2 = f.orb_rigid_alignment(reference_image, inspected_image,
                                                       max_features, good_match_percent, diffy_thresh, data_path)

# fix the transformation matrix from orb_rigid_alignment to fit ECC_rigid_alignment
for i in range(2):
    for j in range(3):
        h_matrix1[i, j] = h_matrix2[i, j]

# align using ECC
Registered_reference_image, h_matrix = f.ECC_rigid_alignment(inspected_image, reference_image,
                                                                             number_of_iterations,
                                                                             termination_eps, data_path, h_matrix1)

# Since I registered the reference, now the images don't have the same boundaries.
# To compensate for this, let all non-overlapping parts be white in the inspected figure and black in the
# reference figure, to give indication as to which parts of the image were not actually inspected.
inspected_image[Registered_reference_image == 0] = 0

# Calculate the difference between the two images
diff_image = inspected_image.copy()
cv2.absdiff(inspected_image, Registered_reference_image, diff_image)  # The difference is returned in the third argument
# convert the difference into grayscale images
diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

# threshold the gray image to mark clusters of differences.
# This is done using a combination of a strict threshold (could be further adjusted once alignment is optimized)
# and Otsu's threshold, which determins an optimal global threshold value from the image histogram.
strict_thresh = 40
(T, thresholded_strict) = cv2.threshold(diff_gray, strict_thresh, 255, cv2.THRESH_BINARY)
thresholded_Otsu = f.Otsus_thresholding(diff_gray)
# the thresholded map would be the intersection of the strict-thresholded and Otsu-thresholded maps
result = cv2.bitwise_and(thresholded_strict, thresholded_Otsu)

# show process
fig, axs = plt.subplots(1, 4, figsize=(15, 6))
for ax, im in zip(axs, ['reference_image', 'Registered_reference_image', 'inspected_image', 'result']):
    ax.imshow(eval(im))
    ax.set_title(im)
plt.show()

# show final result
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
for ax, im in zip(axs, ['inspected_image', 'result']):
    ax.imshow(eval(im))
    ax.set_title(im)
plt.show()

# Write aligned image to disk
outFilename = "reference_image_aligned.jpg"
print("Saving aligned reference image : ", outFilename)
cv2.imwrite(os.path.join(data_path, outFilename), Registered_reference_image)
outFilename = "inspected_image_new_boundaries.jpg"
print("Saving new inspected image : ", outFilename)
cv2.imwrite(os.path.join(data_path, outFilename), inspected_image)
