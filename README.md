# Spot the differences
Register a flawless reference image to a corrupted image to detect small defects.

Implemented in Python using the OpenCV package.

## How to use
To run, start with Spot_the_differences_MAIN.
You should have the images in .jpg format, under the file names 'inspected_image' and 'reference_image'. Example images are supplied.

Please note that upon running the script, you will be asked to enter the path where the images are stored.

## Pipeline
1.	Alignment of the reference image to the inspected image in 2 stages:

    a.	Rough alignment using a feature-based approach (ORB, Oriented FAST and rotated BRIEF), to compensate 
    for large mismatches.

    b.	Fine alignment using a correlation maximization approach (ECC, Enhanced Correlation Coefficient Maximization).

2.	Calculating the absolute difference between the aligned images.

3.	Thresholding. Since we are interested in detection of small differences, I combined 2 thresholding methods and took their intersection to eliminate some of the random noises.

    a.	Strict threshold, namely a predefined cut off. 

    b.	Otsu's threshold, a method determining an optimal global threshold value from the image histogram, and is therefore specific per image.
