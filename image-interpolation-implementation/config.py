import numpy as np
import cv2

class BilinearInterpolation:

    def __init__(self, image, scale_size:list) -> None:
        if isinstance(image, str):
            self.IMAGE = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  
        elif isinstance(image, np.ndarray):
            self.IMAGE = image
        else:
            raise ValueError('image must be an ndarray or path to image data not {}'.format(type(image)))

        self.SCALE_SIZE = scale_size

    def create_empty_image(self):
        """creates an empty image using scale provided

        Returns:
            ndarray: 2D array of new image
        """
        new_image = np.zeros((self.SCALE_SIZE[0]*self.IMAGE.shape[0], \
                                self.SCALE_SIZE[1]*self.IMAGE.shape[1], 3 if len(self.IMAGE.shape) == 3 else None), dtype='uint8')
        return new_image
        
    def check_and_transform_rgb(self, image):
        """Transforms and scales rgb color channels

        Args:
            image ndarray: a numpy array of image

        Returns:
            ndarray: a numpy array of transformed image channels
        """
        if len(image) == 3:
            r = self.core_transform(image[:,:,0])
            g = self.core_transform(image[:,:,1])
            b = self.core_transform(image[:,:,2])
            channel = r,g,b
            return np.asarray(channel)
        return image

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

    def create_scaler(self, scale:list):
        size = len(scale)
        matrix = np.zeros((size, size))
        for i, j in enumerate(matrix):
            matrix[i][i] = scale[i]
        return matrix

    def core_transform(self):
        new_image = self.create_empty_image()
        scale = self.create_scaler(self.SCALE_SIZE)
        inverse_scale = np.linalg.inv(scale)
        image = self.check_and_transform_rgb(self.IMAGE)

        for i in range(self.IMAGE.shape[0]):
                for j in range(self.IMAGE.shape[1]):
                    point = np.array([i,j])
                    dot_point = inverse_scale.dot(point)
                    new_i, new_j = dot_point
                    if i < 0 or i>=self.IMAGE.shape[0] or j<0 or j>=self.IMAGE.shape[1]:
                        pass
                    else:
                        result = self.apply_bilinear_transform(new_i,new_j,self.IMAGE)
                        new_image[i,j] = result
        return new_image
    


# image = cv2.imread('images/business.jpg')
# bilinear = BilinearInterpolation('images/business.jpg', [2,2])
# print(bilinear.core_transform())