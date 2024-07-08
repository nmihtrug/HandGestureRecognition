import cv2
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt

class HandRecognition:
    def __init__(self):
        self.features_map = pickle.load(open('features/features_map.pkl', 'rb'))

    def _conv(self, image, kernel):

        img = image.copy()
    
        k_H, k_W = kernel.shape
        
        img = np.pad(img, (k_H - 1, k_W - 1), 'edge')
        
        rows, cols = image.shape

        output_image = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                # Extract the region of interest
                region = img[i:i+k_H, j:j+k_W]
                # Apply the kernel (element-wise multiplication and sum the result)
                output = np.sum(region * kernel)
                # Store the result in the output image
                output_image[i, j] = output

        # Normalize the output image to be in the range [0, 255]
        output_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX)
        return output_image

    def _pooling(self, image, size, type='max'):
        # Get image dimensions
        rows, cols = image.shape

        # Define the size of the pooling window
        pool_size = size

        # Calculate the dimensions of the output image
        pooled_rows = rows // pool_size
        pooled_cols = cols // pool_size

        # Create an empty array to store the pooled result
        pooled_image = np.zeros((pooled_rows, pooled_cols), dtype=np.uint8)

        if type == 'max':
            # Apply max pooling
            for i in range(pooled_rows):
                for j in range(pooled_cols):
                    # Define the region of interest
                    region = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                    # Get the maximum value in the region
                    max_value = np.max(region)
                    # Store the result in the pooled image
                    pooled_image[i, j] = max_value
        elif type == 'min':
            # Apply min pooling
            for i in range(pooled_rows):
                for j in range(pooled_cols):
                    # Define the region of interest
                    region = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                    # Get the minimum value in the region
                    min_value = np.min(region)
                    # Store the result in the pooled image
                    pooled_image[i, j] = min_value
        return pooled_image

    def _feature_extract(self, image, type = 1, show=False):
        kernel1 = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]])

        kernel2 = np.array([[1, 0, -1], 
                            [2, 0, -2], 
                            [1, 0, -1]])

        kernel3 = np.array([[1, 2, 1], 
                            [0,  0, 0], 
                            [-1, -2, -1]])
        
        
        kernel4 = np.array([[-1, -2, -1], 
                            [0,  0, 0], 
                            [1, 2, 1]])

        if type == 1:
            kernel = kernel1
        elif type == 2:
            kernel = kernel2
        elif type == 3:
            kernel = kernel3
        elif type == 4:
            kernel = kernel4

        output = []

        out = image.copy()
        output.append(out)

        out = self._conv(out, kernel)
        output.append(out)

        out = self._pooling(out, 2, "max")
        output.append(out)

        out = self._pooling(out, 2, "max")
        output.append(out)

        out = self._pooling(out, 2, "max")
        output.append(out)

        if show:
            fig, ax = plt.subplots(1, len(output), figsize=(15, 15 * len(output)))
            fig.tight_layout()
            for ax, img in zip(ax, output):
                ax.imshow(img, cmap="gray")
                ax.title.set_text(img.shape)
            plt.show()

        return out

    def _fusion_feature(self, image):
        fusion_feature = np.array([])
        for i in range(1, 5):
            feature = self._feature_extract(image, i)
            feature = feature.flatten()
            
            fusion_feature = np.concatenate([fusion_feature, feature])
        
        # Normalize
        fusion_feature = (fusion_feature - fusion_feature.min()) / (fusion_feature.max() - fusion_feature.min()) 
        return np.array(fusion_feature)


    def _cosine_similarity(self, a, b):
        return (a @ b.T) / (np.linalg.norm(a)*np.linalg.norm(b))

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def predict(self, image, mode = 'ConSim'):
        fusion_feature = self._fusion_feature(image)
        pred = -1
        
        if mode == 'MLP':
            fusion_feature = torch.from_numpy(fusion_feature).float()
            fusion_feature = fusion_feature.reshape(1, -1)
            output = self.MLPmodel(fusion_feature)
            pred = torch.argmax(output, dim = 1).item()
            
        elif mode == 'ConSim':
            cos_sim_map = np.zeros(10)
            for i in range(10):
                cos_sim = self._cosine_similarity(self.features_map[i], fusion_feature)
                cos_sim_map[i] = cos_sim
            pred = cos_sim_map.argmax()
            
        elif mode == 'EuDis':
            euc_dis_map = np.zeros(10)
            for i in range(10):
                euc_dis = self._euclidean_distance(self.features_map[i], fusion_feature)
                euc_dis_map[i] = euc_dis
            pred = euc_dis_map.argmin()
            
        else:
            Exception('Invalid mode')
        return pred