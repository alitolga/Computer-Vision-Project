"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys

import numpy as np
import time


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def switchToGrayscale(_frame):
    frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    return frame

def gaussianFilter(_frame, _ksize = (5,5), _sigmaX = 0):
    ksize = _ksize
    sigmaX = _sigmaX
    frame = cv2.GaussianBlur(_frame, ksize, sigmaX)
    return frame

def bilateralFilter(_frame, _diameter = 9, _sigmaColor = 75, _sigmaSpace = 75):
    diameter = _diameter
    sigmaColor = _sigmaColor
    sigmaSpace = _sigmaSpace
    frame = cv2.bilateralFilter(frame, diameter, sigmaColor, sigmaSpace)
    return frame

def showRGBandHSVmasks(frame, cap):
    # Convert the image to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the blue color range in both RGB and HSV color space
    blue_range_rgb = np.array([70, 0, 0]), np.array([255, 75, 75])
    # blue_range_rgb = np.array([100, 0, 0]), np.array([255, 100, 100])

    # blue_range_hsv = np.array([100, 150, 0]), np.array([140, 255, 255])
    blue_range_hsv = np.array([110, 50, 70]), np.array([130, 255, 255])

    # Create binary masks for the object in both color spaces
    mask_rgb = cv2.inRange(frame, blue_range_rgb[0], blue_range_rgb[1])
    mask_hsv = cv2.inRange(hsv_frame, blue_range_hsv[0], blue_range_hsv[1])

    if between(cap, 4000, 8000):
        # Apply morphological operations to improve the grabbing
        print("Morphing...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
        mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
        # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
        # mask_hsv = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)

    # Display the results
    cv2.imshow('RGB mask', mask_rgb)
    cv2.imshow('HSV mask', mask_hsv)

def EdgeDetectorSobel(frame):
    # Convert the image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # delta = 100
    delta = 0
    scale = 1

    # Apply the Sobel edge detector with the updated parameters
    sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, scale=scale, delta=delta, ksize=-1)
    sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, scale=scale, delta=delta, ksize=-1)

    edges_horizontal = cv2.convertScaleAbs(sobelx)
    edges_vertical = cv2.convertScaleAbs(sobely)
    # edges = cv2.addWeighted(edges_horizontal, 0.5, edges_vertical, 0.5, 0)
    edges = cv2.magnitude(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Visualize the edges with color
    colored_edges = cv2.merge([np.zeros_like(edges), edges, edges])
    # cv2.imshow('Colored Edges', colored_edges)
    # cv2.imshow('Edges', edges)
    return edges


def PrintHoughCircles(frame):
    edges = EdgeDetectorSobel(frame)

    # Hough transform for circles parameters
    min_radius = 10
    max_radius = 50
    min_distance = 20
    # sobel_kernel_size = 3
    # hough_threshold = 70
    # param1 = 70
    # param2 = 50
    param1 = 50
    param2 = 60

    # Apply Hough transform for circle detection
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=min_distance,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    # # Draw the detected circles on the original color frame
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
        for (x, y, r) in circles:
            # cv2.circle(gray_frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)


def TemplateMatching(frame):
    object_img = cv2.imread('ball_ss.png')
    object_gray = switchToGrayscale(object_img)
    object_h, object_w = object_gray.shape[:2]

    match_method = cv2.TM_CCOEFF_NORMED

    frame = switchToGrayscale(frame)

    result = cv2.matchTemplate(frame, object_gray, match_method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # print(min_val, max_val, min_loc, max_loc)

    if (match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED):
        match_loc = min_loc
    else:
        match_loc = max_loc

    certainty_color = 255 * max_val
    # print(certainty_color)

    cv2.rectangle(frame, match_loc, (match_loc[0] + object_h, match_loc[1] + object_w),
                  (certainty_color, certainty_color, certainty_color), -1)


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)

            # if between(cap, 500, 1000):
            #     frame = switchToGrayscale(frame)
            # if between(cap, 1500, 2000):
            #     frame = switchToGrayscale(frame)
            # if between(cap, 2500, 3000):
            #     frame = switchToGrayscale(frame)
            # if between(cap, 3500, 4000):
            #     frame = switchToGrayscale(frame)
            #
            # if between(cap, 4000, 5000):
            #     print("Gaussian filter 1")
            #     frame = gaussianFilter(frame, (5, 5), 0)
            #     # frame = cv2.bilateralFilter(frame, 9, 75, 75)
            #
            # if between(cap, 5000, 6000):
            #     print("Gaussian filter 2")
            #     frame = cv2.GaussianBlur(frame, (5, 5), 1)
            #     # frame = cv2.bilateralFilter(frame, 9, 10, 50)
            #
            # if between(cap, 6000, 7000):
            #     print("Bilateral filter 1")
            #     frame = cv2.bilateralFilter(frame, 9, 250, 50)
            #     # frame = cv2.GaussianBlur(frame, (5, 5), 2)
            #
            # if between(cap, 7000, 8000):
            #     print("Bilateral filter 2")
            #     frame = cv2.bilateralFilter(frame, 9, 10, 250)
            #     # frame = cv2.GaussianBlur(frame, (5, 5), 3)
            #
            # if between(cap, 0, 8000):
            #     showRGBandHSVmasks(frame, cap)

            # if between(cap, 0, 8000):
            # edges = EdgeDetectorSobel(frame)

            # if between(cap, 0, 8000):
            #     TemplateMatching(frame)


            """ write frame that you processed to output"""
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)