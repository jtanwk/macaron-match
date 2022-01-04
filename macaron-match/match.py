import os
import argparse
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Matcher:
    raw_path: str
    out_path: str
    line_color: Tuple[int]
    line_thickness: int
    im_raw: np.ndarray
    im_rgb: np.ndarray
    im_hsv: np.ndarray
    im_gray: np.ndarray
    im_filt: np.ndarray
    im_canny: np.ndarray
    im_closed: np.ndarray
    all_contours: List[np.ndarray]
    filt_contours: List[np.ndarray]
    ordered_contours: List[np.ndarray]

    def __init__(self, raw_path: str, out_path: str):
        self.raw_path = raw_path
        self.out_dir = out_path

        # Plotting/drawing defaults
        self.line_color = (255, 0, 0)
        self.line_thickness = 2
        self.line_type = cv2.LINE_AA
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.figsize = (16, 16)

    def match(self):

        (
            self.read_image()
            .preprocess_color_space()
            .get_highest_contrast_channel()
            .apply_bilateral_filter()
            .detect_edges()
            .close_open_edges()
            .find_contours()
            .draw_contours(self.all_contours, "01-all-contours.png")
            .filter_contours()
            .draw_contours(self.filt_contours, "02-final-contours.png", num=True)
            .sort_and_pair()
            .draw_paired_contours("03-final-matches.png")
        )

    def read_image(self):
        if not os.path.isfile(self.raw_path):
            raise FileNotFoundError(f"File not found at {self.raw_path}")
        self.im_raw = cv2.imread(self.raw_path)
        return self

    def preprocess_color_space(self):
        self.im_rgb = cv2.cvtColor(self.im_raw, cv2.COLOR_BGR2RGB)
        self.im_hsv = cv2.cvtColor(self.im_rgb, cv2.COLOR_RGB2HSV)
        return self

    def get_highest_contrast_channel(self):
        channels = cv2.split(self.im_hsv)
        contrasts = [x.std() for x in channels]
        idx_max = np.argmax(contrasts)
        print(f"Highest contrast is channel {idx_max}: {contrasts[idx_max]}")
        self.im_gray = channels[idx_max]
        return self

    def apply_bilateral_filter(
        self, d: int = 5, sigma_color: int = 175, sigma_space: int = 175
    ):
        self.im_filt = cv2.bilateralFilter(self.im_gray, d, sigma_color, sigma_space)
        return self

    def detect_edges(self, min_val: int = 100, max_val: int = 150):
        self.im_canny = cv2.Canny(self.im_filt, min_val, max_val)
        return self

    def close_open_edges(
        self, kernel_size: Tuple[int] = (2, 2), num_iterations: int = 1
    ):
        kernel = np.ones(kernel_size, np.uint8)
        self.im_closed = cv2.morphologyEx(
            self.im_canny, cv2.MORPH_CLOSE, kernel, iterations=num_iterations
        )
        return self

    def find_contours(self):
        self.all_contours, _ = cv2.findContours(
            self.im_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"Detected {len(self.all_contours)} contours")
        return self

    def filter_contours(self, top_n: int = 30, min_points: int = 8):

        # Minimum area = half of mean area of top N contours
        cnt_areas = [cv2.contourArea(x) for x in self.all_contours]
        cnt_areas_top_n = sorted(cnt_areas)[-top_n:]
        cnt_areas_median = cnt_areas_top_n[top_n // 2]
        min_area = 0.5 * cnt_areas_median

        # Keep only circular contours above min_area and min_points
        circles = []
        for i in self.all_contours:

            # Approximate contours as simple polygon
            epsilon = 0.01 * cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, epsilon, closed=True)

            # Keep only if approximate polygon is like a circle
            area = cv2.contourArea(i)
            if len(approx) > min_points and area > min_area:
                circles.append(i)

        self.filt_contours = circles
        print(f"Kept {len(self.filt_contours)} of {len(self.all_contours)} contours")
        return self

    def draw_contours(self, cnt: List[np.ndarray], out_name: str, num: bool = False):

        # Draw contours
        plt.figure(figsize=self.figsize)
        im_temp = self.im_rgb.copy()
        cv2.drawContours(im_temp, cnt, -1, self.line_color, self.line_thickness)

        # Label each contour with a number
        if num:
            for idx, i in enumerate(cnt):

                # Compute contour centroid
                m = cv2.moments(i)
                x = int(m["m10"] / m["m00"])
                y = int(m["m01"] / m["m00"])

                # Label text
                cv2.putText(
                    im_temp,
                    text=str(idx + 1),
                    org=(x, y),
                    fontFace=self.font_face,
                    fontScale=self.font_scale,
                    color=self.line_color,
                    thickness=self.line_thickness,
                    lineType=self.line_type,
                )

        # Plot and save figure
        output_name = os.path.join(self.out_dir, out_name)
        plt.imshow(im_temp)
        plt.savefig(output_name)

        return self

    def sort_and_pair(self):
        sorted_contours = sorted(self.filt_contours, key=lambda x: cv2.contourArea(x))
        odd_idx = list(range(0, len(self.filt_contours) - 1, 2))
        self.ordered_contours = [
            (sorted_contours[x], sorted_contours[x + 1]) for x in odd_idx
        ]
        print(f"{len(odd_idx) * 2} of {len(sorted_contours)} contours paired")
        return self

    def draw_paired_contours(self, out_name: str):

        plt.figure(figsize=self.figsize)
        im_temp = self.im_rgb.copy()
        cmap = matplotlib.cm.get_cmap("Dark2")

        for idx, i in enumerate(self.ordered_contours):

            # Draw circle
            r, g, b, a = (int(x) for x in cmap(idx % cmap.N, bytes=True))
            line_color = (r, g, b)
            cv2.drawContours(im_temp, i, -1, line_color, thickness=cv2.FILLED)

            for j in i:
                # Label with pair idx
                m = cv2.moments(j)
                x = int(m["m10"] / m["m00"])
                y = int(m["m01"] / m["m00"])
                cv2.putText(
                    im_temp,
                    text=str(idx + 1),
                    org=(x, y),
                    fontFace=self.font_face,
                    fontScale=self.font_scale,
                    color=(255, 255, 255),
                    thickness=self.line_thickness,
                    lineType=self.line_type,
                )

        output_name = os.path.join(self.out_dir, out_name)
        plt.imshow(im_temp)
        plt.savefig(output_name)
        print(f"Matches saved to f{output_name}")

        return self


if __name__ == "__main__":

    # Get inputs from user
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-path", dest="raw_path", help="filepath to input image"
    )
    parser.add_argument(
        "-o", "--output-path", dest="out_path", help="Directory to save outputs to"
    )
    args = parser.parse_args()

    # Match
    Matcher(args.raw_path, args.out_path).match()
