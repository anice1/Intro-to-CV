import numpy as np
import cv2
import matplotlib.pyplot as plt

class BilinearInterpolation:

    def __init__(self, image, scale_size) -> None:
        self.IMAGE = image
        self.SCALE_SIZE = scale_size

        self.IMAGE = self.read_image()

    def read_image(self):
        """Transforms image data to numpy array

        Returns:
            ndarray: 2D array of given image
        """
        return cv2.imread(self.IMAGE, cv2.IMREAD_GRAYSCALE)

    def create_empty_image(self):
        """creates an empty image using scale provided

        Returns:
            ndarray: 2D array of new image
        """
        new_image = np.zeros((self.SCALE_SIZE*self.IMAGE.shape[0], \
                                self.SCALE_SIZE*self.IMAGE.shape[1]))
        return new_image
        
    def apply_bilinear_transform(self, row, column, image):
        """Applies bilinear interpolation on an image

        Args:
            row (float): pixel current row cordinate
            column (float): pixel curent column cordinate
            image (2D array): transformed image

        Returns:
            A bilinear interpolation result of added intensities
        """
        left_column = int(column)
        right_column = left_column + 1

        distance_to_the_left_col = column - left_column
        distance_to_the_right_col = right_column - column

        top_row = int(row)
        bottom_row = top_row + 1

        distance_to_the_left_row = bottom_row - row
        distance_to_the_right_row = row - top_row

        # Calculate the intensities
        ## Check if we are within image boundary
        if top_row>=0 and bottom_row < self.IMAGE.shape[0] and left_column >=0 and right_column<self.IMAGE.shape[1]:
            column_intensity = (distance_to_the_right_col * self.IMAGE[top_row, left_column]) + (distance_to_the_left_col * self.IMAGE[top_row, right_column])
            row_intensity = (distance_to_the_right_col * self.IMAGE[bottom_row, left_column]) + (distance_to_the_left_col * self.IMAGE[bottom_row, right_column])
            final_intensity = np.uint8((distance_to_the_right_row * column_intensity) + (distance_to_the_left_row * row_intensity))
            return final_intensity
        return 0

    def core_transform(self):
        new_image = self.create_empty_image()
        scale = np.array([[self.SCALE_SIZE,0],[0,self.SCALE_SIZE]])
        inverse_scale = np.linalg.inv(scale)

        for i in range(self.IMAGE.shape[0]):
            for j in range(self.IMAGE.shape[1]):
                point = np.array([i,j])
                dot_point = inverse_scale.dot(point)
                new_i, new_j = dot_point

                result = self.apply_bilinear_transform(i,j,self.IMAGE)
                new_image[i,j] = result
        return new_image


bilinear = BilinearInterpolation('images/image.png', 2)
