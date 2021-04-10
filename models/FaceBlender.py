import numpy as np
import cv2 as cv

from utils import *
import consts
import settings


# NOTE: This class is just a wrapper around 'align_warp_back_multiple_dlib.py'
class FaceBlender:
    @staticmethod
    def calculate_cdf(histogram):
        """
        This method calculates the cumulative distribution function
        :param array histogram: The values of the histogram
        :return: normalized_cdf: The normalized cumulative distribution function
        :rtype: array
        """
        # Get the cumulative sum of the elements
        cdf = histogram.cumsum()

        # Normalize the cdf
        normalized_cdf = cdf / float(cdf.max())

        return normalized_cdf

    @staticmethod
    def calculate_lookup(src_cdf, ref_cdf):
        """
        This method creates the lookup table
        :param array src_cdf: The cdf for the source image
        :param array ref_cdf: The cdf for the reference image
        :return: lookup_table: The lookup table
        :rtype: array
        """
        lookup_table = np.zeros(256)
        lookup_val = 0
        for src_pixel_val in range(len(src_cdf)):
            lookup_val
            for ref_pixel_val in range(len(ref_cdf)):
                if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                    lookup_val = ref_pixel_val
                    break
            lookup_table[src_pixel_val] = lookup_val
        return lookup_table

    @staticmethod
    def match_histograms(src_image, ref_image):
        """
        This method matches the source image histogram to the
        reference signal
        :param image src_image: The original source image
        :param image  ref_image: The reference image
        :return: image_after_matching
        :rtype: image (array)
        """
        # Split the images into the different color channels
        # b means blue, g means green and r means red
        src_b, src_g, src_r = cv.split(src_image)
        ref_b, ref_g, ref_r = cv.split(ref_image)

        # Compute the b, g, and r histograms separately
        # The flatten() Numpy method returns a copy of the array c
        # collapsed into one dimension.
        src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
        src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
        src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
        ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
        ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
        ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

        # Compute the normalized cdf for the source and reference image
        src_cdf_blue = FaceBlender.calculate_cdf(src_hist_blue)
        src_cdf_green = FaceBlender.calculate_cdf(src_hist_green)
        src_cdf_red = FaceBlender.calculate_cdf(src_hist_red)
        ref_cdf_blue = FaceBlender.calculate_cdf(ref_hist_blue)
        ref_cdf_green = FaceBlender.calculate_cdf(ref_hist_green)
        ref_cdf_red = FaceBlender.calculate_cdf(ref_hist_red)

        # Make a separate lookup table for each color
        blue_lookup_table = FaceBlender.calculate_lookup(src_cdf_blue, ref_cdf_blue)
        green_lookup_table = FaceBlender.calculate_lookup(src_cdf_green, ref_cdf_green)
        red_lookup_table = FaceBlender.calculate_lookup(src_cdf_red, ref_cdf_red)

        # Use the lookup function to transform the colors of the original
        # source image
        blue_after_transform = cv.LUT(src_b, blue_lookup_table)
        green_after_transform = cv.LUT(src_g, green_lookup_table)
        red_after_transform = cv.LUT(src_r, red_lookup_table)

        # Put the image back together
        image_after_matching = cv.merge([blue_after_transform, green_after_transform, red_after_transform])
        image_after_matching = cv.convertScaleAbs(image_after_matching)

        return image_after_matching

    @staticmethod
    def blur_blending(im1, im2, mask):
        mask *= 255.0
        kernel = np.ones((9, 9), dtype=np.uint8)
        mask = cv.erode(mask, kernel, iterations=3)
        mask_blur = cv.GaussianBlur(mask, (25, 25), 0)
        mask_blur /= 255.0

        im = im1 * mask_blur + (1 - mask_blur) * im2

        im /= 255.0
        im = np.clip(im, 0.0, 1.0)

        return im * 255.


    def __call__(self, image, faces_with_affines, enhanced_faces):
        assert len(faces_with_affines) == len(enhanced_faces)

        for face_id, (face_image, landmarks_affine, inverse_landmarks_affine) in enumerate(faces_with_affines):
            # Histogram color matching between enhanced image and enchanced face image
            A = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            B = cv.cvtColor(enhanced_faces[face_id], cv.COLOR_RGB2BGR)
            B = FaceBlender.match_histograms(B, A)
            enhanced_face = cv.cvtColor(B, cv.COLOR_BGR2RGB)

            # Create mask on where to blend 'enhanced_face' in 'image'
            mask = cv.warpAffine(np.ones_like(image, dtype=np.uint8), inverse_landmarks_affine, face_image.shape[0:2])

            image = FaceBlender.blur_blending(enhanced_face, image, mask.astype(np.float32))

            #backward_mask = cv.warpAffine(np.ones_like(image, dtype=np.uint8), landmarks_affine, face_image.shape[0:2])

            ## Histogram color matching
            #A = cv.cvtColor(face_image, cv.COLOR_RGB2BGR)
            #B = cv.cvtColor(enhanced_faces[face_id], cv.COLOR_RGB2BGR)
            #B = FaceBlender.match_histograms(B, A)
            #enhanced_face = cv.cvtColor(B, cv.COLOR_BGR2RGB)

            #enhanced_face = cv.warpAffine(enhanced_face, inverse_landmarks_affine, image.shape[0:2])
            #backward_mask = cv.warpAffine(backward_mask, inverse_landmarks_affine, image.shape[0:2])

            #image = FaceBlender.blur_blending(enhanced_face, image, backward_mask)

        return image