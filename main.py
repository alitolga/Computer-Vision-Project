"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys

import numpy as np


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def switch_between_color_and_grayscale(frame):
    # Switch mode
    # print(type(frame))
    if frame.channels() == 3:
        _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print("Here")
        # _mode = 'grayscale'
    else:
        _frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # _mode = 'color'
    return _frame


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    mode = 'color'

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            frame = cv2.flip(frame, 0)

            # if between(cap, 500, 1000):
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if between(cap, 1500, 2000):
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if between(cap, 2500, 3000):
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if between(cap, 3500, 4000):
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # cv2.waitKey(1000)
            # print(mode)
            # pass
            # ...

            # if between(cap, 4000, 5000):
            #     print("Gaussian filter 1")
            #     # frame = cv2.GaussianBlur(frame, (5, 5), 0)
            #     frame = cv2.bilateralFilter(frame, 9, 75, 75)
            #
            # if between(cap, 5000, 6000):
            #     print("Gaussian filter 2")
            #     frame = cv2.GaussianBlur(frame, (5, 5), 1)
            #     frame = cv2.bilateralFilter(frame, 9, 10, 50)
            #
            # if between(cap, 6000, 7000):
            #     print("Gaussian filter 3")
            #     frame = cv2.GaussianBlur(frame, (5, 5), 2)
            #     frame = cv2.bilateralFilter(frame, 9, 250, 50)
            #
            # if between(cap, 10000, 12000):
            #     print("Gaussian filter 4")
            #     # frame = cv2.GaussianBlur(frame, (5, 5), 3)
            #     frame = cv2.bilateralFilter(frame, 9, 10, 250)

            # Convert the image to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the blue color range in both RGB and HSV color space
            blue_range_rgb = np.array([70, 0, 0]), np.array([255, 75, 75])
            # blue_range_rgb = np.array([100, 0, 0]), np.array([255, 100, 100])

            # blue_range_hsv = np.array([100, 150, 0]), np.array([140, 255, 255])
            # blue_range_hsv = np.array([110,50,70]), np.array([130,255,255])
            blue_range_hsv = np.array([110,50,70]), np.array([130,255,255])

            # Create binary masks for the object in both color spaces
            mask_rgb = cv2.inRange(frame, blue_range_rgb[0], blue_range_rgb[1])
            mask_hsv = cv2.inRange(hsv_frame, blue_range_hsv[0], blue_range_hsv[1])

            # Apply morphological operations to improve the grabbing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
            mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
            # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
            # mask_hsv = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)

            # Display the results
            cv2.imshow('RGB mask', mask_rgb)
            cv2.imshow('HSV mask', mask_hsv)

            # Wait for the user to close the windows
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            """
            # Apply Gaussian filter
            gaussian = cv2.GaussianBlur(frame, (5, 5), 0)
        
            # Apply bilateral filter
            bilateral = cv2.bilateralFilter(frame, 9, 75, 75)
            """

            # write frame that you processed to output
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