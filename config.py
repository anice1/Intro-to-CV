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
                                self.SCALE_SIZE[1]*self.IMAGE.shape[1]))
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

image = cv2.imread('images/business.jpg', cv2.IMREAD_GRAYSCALE)
bilinear = BilinearInterpolation('images/business.jpg', [1,1])
angle = 45 
cos = np.cos(np.deg2rad(angle))
sin = np.sin(np.deg2rad(angle))

scale = np.array([[cos, -sin],[sin, cos]])
row, col = image.shape[:2]
cordinate = np.array([[0,0],[0,col-1],[row-1,0],[row-1,col-1]])
dot_cord = scale.dot(cordinate.T)

minimums = dot_cord.min(axis=1)
maximums = dot_cord.max(axis=1)

min_row = np.int64(np.floor(minimums[0]))
min_col = np.int64(np.floor(minimums[1]))

max_row = np.int64(np.floor(maximums[0]))
max_col = np.int64(np.floor(maximums[1]))

row = max_row - min_row +1
col = max_col - min_col +1

# Create empty image 
empty_image = np.zeros((row, col), dtype=np.uint8)
scale_inverse = np.linalg.inv(scale) # Index shifting

for i in range(min_row, max_row):
    for j in range(min_col, max_col):
        point = np.array([i,j])
        dot_point = scale_inverse.dot(point)
        new_i, new_j = dot_point
        if i<0 or i>=image.shape[0] or j<0 or j>=image.shape[1]:
            pass
        else:
            g = bilinear.apply_bilinear_transform(new_i, new_j, image)
            empty_image[i-min_row, j-min_col] = g

