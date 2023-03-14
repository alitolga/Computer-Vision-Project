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

    # Sobel Edge detection parameters
    scale = 1
    delta = 0

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

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)

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
            # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #
            # # Define the blue color range in both RGB and HSV color space
            # blue_range_rgb = np.array([70, 0, 0]), np.array([255, 75, 75])
            # # blue_range_rgb = np.array([100, 0, 0]), np.array([255, 100, 100])
            #
            # # blue_range_hsv = np.array([100, 150, 0]), np.array([140, 255, 255])
            # blue_range_hsv = np.array([110,50,70]), np.array([130,255,255])
            #
            # # Create binary masks for the object in both color spaces
            # mask_rgb = cv2.inRange(frame, blue_range_rgb[0], blue_range_rgb[1])
            # mask_hsv = cv2.inRange(hsv_frame, blue_range_hsv[0], blue_range_hsv[1])
            #
            # # Apply morphological operations to improve the grabbing
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
            # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
            # mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
            # # mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
            # # mask_hsv = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel)
            #
            # # Display the results
            # cv2.imshow('RGB mask', mask_rgb)
            # cv2.imshow('HSV mask', mask_hsv)







            # Convert the image to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply the Sobel edge detector for horizontal edges
            # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            # edges_horizontal = cv2.convertScaleAbs(sobelx)
            #
            # # Apply the Sobel edge detector for vertical edges
            # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            # edges_vertical = cv2.convertScaleAbs(sobely)

            # Merge the horizontal and vertical edges
            # edges = cv2.addWeighted(edges_horizontal, 0.5, edges_vertical, 0.5, 0)

            # Display the edges for 5 seconds, and tweak the Sobel detector parameters
            # for scale in np.linspace(0.1, 2, num=20):
            #     for delta in np.linspace(0, 255, num=20):
            # delta = 100

            # if between(cap, 0, 20000):

            # Apply the Sobel edge detector with the updated parameters
            sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, scale=scale, delta=delta, ksize=-1)
            sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, scale=scale, delta=delta, ksize=-1)

            # edges_horizontal = cv2.convertScaleAbs(sobelx * scale + delta)
            edges_horizontal = cv2.convertScaleAbs(sobelx)
            # edges_vertical = cv2.convertScaleAbs(sobely * scale + delta)
            edges_vertical = cv2.convertScaleAbs(sobely)
            # edges = cv2.addWeighted(edges_horizontal, 0.5, edges_vertical, 0.5, 0)
            edges = cv2.magnitude(sobelx, sobely)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Visualize the edges with color
            colored_edges = cv2.merge([np.zeros_like(edges), edges, edges])
            # cv2.imshow('Edges', colored_edges)
            # cv2.imshow('Edges', edges)
            cv2.waitKey(50)  # wait for 50ms
            # delta += 1
            # print(delta)
            # scale += 0.01
            # print(scale)

            # Apply Hough transform for circle detection
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=min_distance,
                                           param1=param1, param2=param2,
                                           minRadius=min_radius, maxRadius=max_radius)

            # Draw the detected circles on the original color frame
            if circles is not None:
                circles = np.round(circles[0, :]).astype('int')
                for (x, y, r) in circles:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

                # Create a mask to highlight the object of interest with a flashy rectangle
                obj_x, obj_y, obj_w, obj_h = x - r, y - r, r * 2, r * 2

                frame_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                frame_mask[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w] = 255
                # rect_img = cv2.rectangle(frame.copy(), (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), (0, 0, 255), 2)
                masked_img = cv2.addWeighted(frame, 0.7, cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

                # Display the rectangle image for 2 seconds
                # cv2.imshow('Normal Rectangle', rect_img)
                cv2.imshow('Flashy Rectangle', masked_img)
                # cv2.waitKey(2000)

                # Create a grayscale map of the likelihood of the object of interest being at each location
                gray_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                # for y in range(gray_map.shape[0]):
                #     for x in range(gray_map.shape[1]):

                # Compute the feature representation of the current image patch (just use the grayscale intensity values)
                # patch_feature = gray_frame[y:y + obj_h, x:x + obj_w]
                # print(x,y,obj_x,obj_y,obj_w,obj_h)

                # Compute the mean squared error between the feature representations
                mse = np.mean((edges - gray_frame) ** 2)
                # mse = np.mean((gray_frame[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w] - patch_feature) ** 2)

                # Invert the error value to get the likelihood
                likelihood = 255 - int(255 * (mse / np.max(mse)))

                # Set the grayscale map value at the current location to the likelihood
                gray_map[y, x] = likelihood


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