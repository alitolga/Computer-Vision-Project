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


def gaussianFilter(_frame, _ksize=(5, 5), _sigmaX=0):
    ksize = _ksize
    sigmaX = _sigmaX
    frame = cv2.GaussianBlur(_frame, ksize, sigmaX)
    return frame


def bilateralFilter(_frame, _diameter=9, _sigmaColor=75, _sigmaSpace=75):
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if between(cap, 14000, 16000):
        # Apply morphological operations to improve the grabbing
        print("Morphing RGB...")
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
        # mask_hsv = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)

    if between(cap, 18000, 20000):
        print("Morphing HSV...")
        # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
        mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
        mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)

    # Display the results
    cv2.imshow('RGB mask', mask_rgb)
    cv2.imshow('HSV mask', mask_hsv)
    if between(cap, 12000, 16000):
        return mask_rgb
    return mask_hsv


def EdgeDetectorSobel(frame, cap):
    # Convert the image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # delta = 100
    delta = 0
    scale = 1
    # sobel_kernel_size = 3

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
    cv2.imshow('Colored Edges', colored_edges)
    cv2.imshow('Edges', edges)
    if between(cap, 20000, 22000):
        return colored_edges
    return edges


def PrintHoughCircles(frame, cap, _min_radius=10, _max_radius=50, _min_distance=20, _param1=50, _param2=60):
    edges = EdgeDetectorSobel(frame, cap)

    # Hough transform for circles parameters
    min_radius = _min_radius
    max_radius = _max_radius
    min_distance = _min_distance
    # param1 = 70
    # param2 = 50
    param1 = _param1
    param2 = _param2

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
    return frame


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

    return frame


def ReplaceBallWithImage(frame):
    replacement_image = cv2.imread('binali.png')

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the object color
    mask = cv2.inRange(hsv_frame, np.array([110, 50, 70]), np.array([130, 255, 255]))

    # Find contours of the ball
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and replace the ball with the ball image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        replacement_image_resized = cv2.resize(replacement_image, (w, h))
        frame[y:y + h, x:x + w] = replacement_image_resized

    return frame


def ChangeColorOfBall(frame, x, y, r):
    ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(ball_mask, (x, y), r, (255, 255, 255), -1)
    alpha = cv2.merge((ball_mask, ball_mask, ball_mask, ball_mask))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame[np.where((alpha == [255, 255, 255, 255]).all(axis=2))] = [0, 0, 255, 0]
    return frame

def CopyBallObject(frame, x, y, r):
    copy_frame1 = frame[(y + r): (y + r * 3), (x - r): (x + r)]
    if copy_frame1.shape == frame[(y - r): (y + r), (x - r): (x + r)].shape:
        # frame[(y + r): (y + r * 3), (x - r): (x + r)] = frame[(y - r): (y + r), (x - r): (x + r)]
        frame[(y + r): (y + r * 3), (x - r): (x + r)] = frame[(y - r): (y + r), (x - r): (x + r)]

    copy_frame2 = frame[(y - r * 3): (y - r), (x - r): (x + r)]
    if copy_frame2.shape == frame[(y - r): (y + r), (x - r): (x + r)].shape:
        # frame[(y + r): (y + r * 3), (x - r): (x + r)] = frame[(y - r): (y + r), (x - r): (x + r)]
        frame[(y - r * 3): (y - r), (x - r): (x + r)] = frame[(y - r): (y + r), (x - r): (x + r)]

    return frame


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)

            if between(cap, 0, 1500):
                frame = switchToGrayscale(frame)
            if between(cap, 1700, 2000):
                frame = switchToGrayscale(frame)
            if between(cap, 2500, 3000):
                frame = switchToGrayscale(frame)
            if between(cap, 3500, 4000):
                frame = switchToGrayscale(frame)

            if between(cap, 4000, 5000):
                print("Gaussian filter 1")
                frame = gaussianFilter(frame, (5, 5), 0)
                # frame = cv2.bilateralFilter(frame, 9, 75, 75)

            if between(cap, 5000, 6000):
                print("Gaussian filter 2")
                frame = cv2.GaussianBlur(frame, (5, 5), 1)
                # frame = cv2.bilateralFilter(frame, 9, 10, 50)

            if between(cap, 6000, 7000):
                print("Gaussian filter 3")
                frame = gaussianFilter(frame, (3, 3), 0)
                # frame = cv2.bilateralFilter(frame, 9, 75, 75)

            if between(cap, 7000, 8000):
                print("Gaussian filter 4")
                frame = cv2.GaussianBlur(frame, (3, 3), 2)
                # frame = cv2.bilateralFilter(frame, 9, 10, 50)

            if between(cap, 8000, 9000):
                print("Bilateral filter 1")
                frame = cv2.bilateralFilter(frame, 7, 250, 50)
                # frame = cv2.GaussianBlur(frame, (5, 5), 2)

            if between(cap, 9000, 10000):
                print("Bilateral filter 2")
                frame = cv2.bilateralFilter(frame, 7, 10, 250)
                # frame = cv2.GaussianBlur(frame, (5, 5), 3)

            if between(cap, 10000, 11000):
                print("Bilateral filter 3")
                frame = cv2.bilateralFilter(frame, 9, 50, 50)
                # frame = cv2.GaussianBlur(frame, (5, 5), 2)

            if between(cap, 11000, 12000):
                print("Bilateral filter 4")
                frame = cv2.bilateralFilter(frame, 9, 250, 250)
                # frame = cv2.GaussianBlur(frame, (5, 5), 3)

            if between(cap, 12000, 20000):
                frame = showRGBandHSVmasks(frame, cap)

            if between(cap, 20000, 24900):
                frame = EdgeDetectorSobel(frame, cap)

            if between(cap, 25000, 27000):
                frame = PrintHoughCircles(frame=frame, cap=cap, _min_radius=90)
                print("min radius 90")
            if between(cap, 27000, 29000):
                frame = PrintHoughCircles(frame=frame, cap=cap, _max_radius=20)
                print("max radius 20")
            if between(cap, 29000, 31000):
                frame = PrintHoughCircles(frame=frame, cap=cap, _param1=150)
                print("param1 too high")
            if between(cap, 31000, 33000):
                frame = PrintHoughCircles(frame=frame, cap=cap, _param2=160)
                print("param2 too high")
            if between(cap, 33000, 35000):
                frame = PrintHoughCircles(frame=frame, cap=cap,)
                print("Ideal Hough Circle Parameters")

            if between(cap, 35000, 40000):
                frame = TemplateMatching(frame)
                print("Template Matching")

            if between(cap, 40000, 60000):
                edges = EdgeDetectorSobel(frame, cap)

                circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                           param1=50, param2=60,
                                           minRadius=10, maxRadius=50)

                if circles is None:
                    continue

                circles = np.round(circles[0, :]).astype('int')

                # Loop through all the detected circles
                for (x, y, r) in circles:
                    if between(cap, 40000, 47000):
                        frame = ChangeColorOfBall(frame, x, y, r)
                        print("Change the color of ball")

                    if between(cap, 47000, 54000):
                        frame = ReplaceBallWithImage(frame)
                        print("Replace ball with image")

                    if between(cap, 54000, 59900):
                        frame = CopyBallObject(frame, x, y, r)
                        print("Make 2 copies of the ball")

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
