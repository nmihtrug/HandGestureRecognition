import cv2
import torch
import torch.nn as nn
import numpy as np
import pickle

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Linear(363, 10)

    def forward(self, x):
        logits = self.linear(x)
        return logits


class HandRecognition:
    
    def __init__(self):
        self.features_map = pickle.load(open('data/features_map.pkl', 'rb'))
        self.MLPmodel = MLP()
        self.MLPmodel.load_state_dict(torch.load('model_final.pth'))
        self.MLPmodel.eval()

    def _conv(self, image, kernel):

        img = image.copy()
        # img = np.pad(img, 1, 'edge')
        
        rows, cols = image.shape

        output_image = np.zeros((rows-2, cols-2))
        
        for i in range(rows-2):
            for j in range(cols-2):
                # Extract the region of interest
                region = img[i:i+3, j:j+3]
                # Apply the kernel (element-wise multiplication and sum the result)
                output = np.sum(region * kernel)
                # Store the result in the output image
                output_image[i, j] = output

        # Normalize the output image to be in the range [0, 255]
        output_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX)
        return output_image

    def _maxPooling(self, image, size):
        # Get image dimensions
        rows, cols = image.shape

        # Define the size of the pooling window
        pool_size = size

        # Calculate the dimensions of the output image
        pooled_rows = rows // pool_size
        pooled_cols = cols // pool_size

        # Create an empty array to store the pooled result
        pooled_image = np.zeros((pooled_rows, pooled_cols), dtype=np.uint8)

        # Apply max pooling
        for i in range(pooled_rows):
            for j in range(pooled_cols):
                # Define the region of interest
                region = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                # Get the maximum value in the region
                max_value = np.max(region)
                # Store the result in the pooled image
                pooled_image[i, j] = max_value
                
        return pooled_image

    def _feature_extract(self, image, type = 1):
        kernel1 =  np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        kernel2 =  np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])
        
        kernel3 = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
        
        
        if type == 1:
            kernel = kernel1
        elif type == 2:
            kernel = kernel2
        elif type == 3:
            kernel = kernel3
            
        out = image.copy()
        out = self._conv(out, kernel)        
        out = self._maxPooling(out, 2)        
        out = self._conv(out, kernel)        
        out = self._maxPooling(out, 2)        
        out = self._maxPooling(out, 2)        
        return out
    
    def _fusion_feature(self, image):
        fusion_feature = np.array([])
        for i in range(1, 4):
            feature = self._feature_extract(image, i)
            feature = feature.flatten()
            
            fusion_feature = np.concatenate([fusion_feature, feature])
        
        # Normalize
        fusion_feature = (fusion_feature - fusion_feature.min()) / (fusion_feature.max() - fusion_feature.min()) 
        return np.array(fusion_feature)


    def _cosine_similarity(self, a, b):
        return (a @ b.T) / (np.linalg.norm(a)*np.linalg.norm(b))

    def _euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    def predict(self, image, mode = 'MLP'):
        fusion_feature = self._fusion_feature(image)
        
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