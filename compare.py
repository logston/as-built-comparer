import csv
import glob
import io
import os
import sys

import cv2
import numpy as np
from PIL import Image as PILImage
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity


class Image:
    STD_WIDTH = 2500
    STD_HEIGHT = 1600

    def __init__(self, path, debug=False):
        self.path = path
        self.img = None

        self.debug = debug
        self.debug_img = None
        self.overlay_img = None

    def convert_pdf(self):
        images = convert_from_path(
            pdf_path=self.path,
            dpi=300,
            fmt="jpeg"
        )

        return np.asarray(images[0])

    def load(self):
        self.img = self.convert_pdf()
        if self.debug:
            self.debug_img = self.img.copy()
        self.validate_image()
        x, y, _w, _h, rotation = self.get_contour_info()
        self.rotate(x, y, rotation)

        # After rotation, x and y (center of bounding box have moved.
        # Need to re-find center.
        _x, _y, w, h, _rotation = self.get_contour_info()
        self.resize(w, h)

        x, y, _w, _h, _rotation = self.get_contour_info()
        self.center(x, y)

    def validate_image(self):
        # Rotate image so longer axis is horizontal
        w = self.img.shape[1]
        h = self.img.shape[0]
        if h > w:
            raise ValueError(f'Image {self.path} requires rotation.')

    def write(self):
        cv2.imwrite(self.path + '.jpg', self.img)
        cv2.imwrite(self.path + '.debug.jpg', self.debug_img)
        if self.debug:
            cv2.imwrite(self.path + '.overlay.jpg', self.overlay_img)

    def get_contour_info(self):
        """
        Return dims of outer most box. Assume that this is box is the
        blue print bounding box.
        """
        # Get contours from B&W image
        img_bw = cv2.cvtColor(
            src=self.img,
            code=cv2.COLOR_RGB2GRAY,
        )
        _, img_bw = cv2.threshold(
            src=img_bw,
            thresh=175,
            maxval=255,
            type=cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            image=img_bw,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        # Find area for each contour
        img_area = self.img.shape[0] * self.img.shape[1]
        contour_details = {}
        for cnt in contours:
            # Find convex hull of contour
            hull = cv2.convexHull(points=cnt, hull=False)
            # Find smallest box that fits the convex hull
            (x, y), (w, h), rotation = cv2.minAreaRect(hull)

            if 0.999 < (w * h) / img_area < 1.001:
                # This is the image bounding box
                continue

            area = w * h
            contour_details[(
                area,
                x,
                y,
                w,
                h,
                rotation,
            )] = hull

        # Get contour with largest area
        contour_details_keys = sorted(contour_details.keys())
        largest_contour_details_key = contour_details_keys[-1]
        area, x, y, w, h, rotation = largest_contour_details_key
        cnt = contour_details[largest_contour_details_key]

        # Ensure width is always the largest dimension
        if h > w:
            #w, h = h, w

            # If cv2 found a rectangle that was taller than
            # it was wide, then the rotation value is probably
            # very large (~ 90 or ~ -90 degrees). This needs to be
            # calibrated down to something between -2 and 2 degrees.

            # Ensure rotation is always small (-2 < rotation < 2)
            if -90 < rotation < -2:
                rotation = 90 + rotation
            elif 90 > rotation > 2:
                rotation = 90 - rotation
            else:
                raise ValueError(f'Rotation error for {self.path}: {rotation}')

        x, y, w, h = int(x), int(y), int(w), int(h)

        if self.debug:
            approx = cv2.approxPolyDP(
                curve=cnt,
                epsilon=0.009 * cv2.arcLength(cnt, True),
                closed=True,
            )
            cv2.drawContours(
                image=self.debug_img,
                contours=[approx],
                contourIdx=0,
                color=(0, 0, 255),
                thickness=10,
            )
            cv2.circle(
                img=self.debug_img,
                center=(x, y),
                radius=15,
                color=(0, 0, 255),
                thickness=-1
            )

        return x, y, w, h, rotation

    def rotate(self, x, y, rotation):
        self.img = self._rotate(self.img, x, y, rotation)
        if self.debug:
            self.debug_img = self._rotate(self.debug_img, x, y, rotation)

    def _rotate(self, img, x, y, rotation):
        img_pil = PILImage.fromarray(img)
        mode = img_pil.mode
        img_pil = img_pil.convert('RGBA')
        img_pil = img_pil.rotate(rotation, center=(x, y), expand=1)

        # Fill in black after rotation
        background = PILImage.new('RGBA', img_pil.size, (255,) * 4)
        img_pil = PILImage.composite(img_pil, background, img_pil)
        img_pil = img_pil.convert(mode)
        return np.asarray(img_pil)

    def resize(self, w, h):
        self.img = self._resize(self.img, w, h)
        if self.debug:
            self.debug_img = self._resize(self.debug_img, w, h)

    def _resize(self, img, w, h):
        scale_factor_w = (self.STD_WIDTH / w)
        scale_factor_h = (self.STD_HEIGHT / h)

        new_w = int(scale_factor_w * img.shape[1])
        new_h = int(scale_factor_h * img.shape[0])

        return cv2.resize(
            src=img,
            dsize=(new_w, new_h),
        )

    def center(self, x, y):
        self.img = self._center(self.img, x, y)
        if self.debug:
            self.debug_img = self._center(self.debug_img, x, y)

    def _center(self, img, x, y):
        # x and y are center of box.
        # Need to make center of box also the center of the image
        top = bottom = left = right = 0

        image_midline = img.shape[1] / 2
        if x < image_midline:
            distance_to_side = img.shape[1] - x
            left = distance_to_side - x

        elif x > image_midline:
            distance_to_side = img.shape[1] - x
            right = x - distance_to_side

        image_midline = img.shape[0] / 2
        if y < image_midline:
            distance_to_side = img.shape[0] - y
            top = distance_to_side - y

        elif y > image_midline:
            distance_to_side = img.shape[0] - y
            bottom = y - distance_to_side

        return cv2.copyMakeBorder(
            src=img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def diff(self, other):
        score, gray_self, gray_other = self._diff(self.img, other.img)
        if self.debug:
            self._diff(self.debug_img, other.debug_img)
        return score

    def _diff(self, img, other_img):
        img, other_img = self.scale_cooperatively(img, other_img)
        gray_self = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_other = cv2.cvtColor(other_img, cv2.COLOR_BGR2GRAY)
        if self.debug:
            self.overlay_img = self.overlay(gray_self, gray_other)
        score = structural_similarity(gray_self, gray_other)
        return score, gray_self, gray_other

    def overlay(self, img, other_img):
        alpha = 0.5
        dst = img.copy()
        cv2.addWeighted(
            src1=img,
            alpha=alpha,
            src2=other_img,
            beta=1-alpha,
            gamma=0,
            dst=dst,
        )
        return dst

    def scale_cooperatively(self, img, other_img):
        """
        Add boarders to each image such that both images are of the same size.
        """
        max_w = max(img.shape[1], other_img.shape[1])
        max_h = max(img.shape[0], other_img.shape[0])

        # Scale img
        left = right = int((max_w - img.shape[1]) / 2)
        top = bottom = int((max_h - img.shape[0]) / 2)
        img = cv2.copyMakeBorder(
            src=img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        # Scale other image
        left = right = int((max_w - other_img.shape[1]) / 2)
        top = bottom = int((max_h - other_img.shape[0]) / 2)
        other_img = cv2.copyMakeBorder(
            src=other_img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        return img, other_img


def load_images(dir_path, debug=False):
    pdf_paths = glob.glob(os.path.join(dir_path, '*.pdf'))
    images = []

    print(
        f'Found {len(pdf_paths)} PDFs. '
        f'Approximate load time: {4 * len(pdf_paths)} seconds.',
        file=sys.stderr,
        flush=True,
    )
    print('Loading images ', end='', file=sys.stderr, flush=True)
    for pdf_path in pdf_paths:
        img = Image(pdf_path, debug=debug)
        img.load()
        images.append(img)
        print('.', end='', file=sys.stderr, flush=True)

    print(' Finished.', file=sys.stderr, flush=True)

    return images


def compare(images):
    print(
        f'Comparing {len(images)} images. '
        f'Approximate comparison time: {4 * len(images)} seconds.',
        file=sys.stderr,
        flush=True,
    )
    print('Comparing images ', end='', file=sys.stderr, flush=True)
    comps = {}
    for image_copy1 in images:
        for image_copy2 in images:
            if image_copy1 is image_copy2:
                continue

            key = frozenset((image_copy1.path, image_copy2.path))
            if key in comps:
                continue

            comps[key] = image_copy1.diff(image_copy2)
            if image_copy1.debug:
                image_copy1.write()

            print('.', end='', file=sys.stderr, flush=True)

    print(' Finished.', file=sys.stderr, flush=True)

    return comps


def sort_comparisons(comps):
    return sorted([value] + sorted(key) for key, value in comps.items())


def print_results(comps):
    with io.StringIO() as fp:
        writer = csv.writer(fp)
        writer.writerow(['Score', 'File A', 'File B'])
        writer.writerows(comps)

        print(fp.getvalue())


def main(dir_path, debug=False):
    images = load_images(dir_path, debug)
    comparisons = compare(images)
    comparisons = sort_comparisons(comparisons)
    print_results(comparisons)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('As-built Comparer')
    parser.add_argument(
        'dir',
        help='Directory with PDFs',
    )
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Output debug images to show progress of program. Program will run much slower.',
    )

    args = parser.parse_args()
    main(args.dir)
