import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
from tqdm import tqdm

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class chdSplit:
    def __init__(self, paths, split_size, iterations, bins):
        self.paths = paths
        self.split_size = split_size
        self.iterations = iterations
        self.bins = bins

        # Calculate the size of the rsulting two subsets
        self.size_b = int(len(paths) * split_size)
        self.size_a = len(paths) - self.size_b

    def __load_img(self, path):
        # If the image is a tif file open it with rasterio 
        if '.tif' in path:
            with rasterio.open(path) as src:
                img = src.read()
                # By deafult rasterio puts the bands at the beginning, moving them at the end
                img = np.moveaxis(img, 0, -1) 
        # Otherwise open it with matplotlib
        else:
            img = np.array(Image.open(path))
        
        return img

    def __random_split(self):
        shuffle = np.random.choice(self.paths, size = len(self.paths))
        subset_a = shuffle[0:self.size_a]
        subset_b = shuffle[self.size_a:self.size_a+self.size_b]

        return shuffle, subset_a, subset_b

    def split(self):
        # Create a dataframe to store iterations
        df = pd.DataFrame(columns = ['Iteration', 'Subset-A', 'Subset-B', 'Dissimilarity'])

        for i in tqdm(range(self.iterations)):
            # 1) Get a random split
            shuffle, subset_a, subset_b = self. __random_split()
            
            # 2) Extract K samples from subset A and subset B
            K = np.min([len(subset_a), len(subset_b)]) # K is set to the min size between subset A and subset B
            
            # 3) Calculate cumulative histogram
            hist_a = 0
            hist_b = 0
            img_a = 0
            img_b = 0

            for k in range(K):
                img_a = self.__load_img(subset_a[k])
                hist_a = hist_a + np.histogram(img_a.flatten(), bins = self.bins)[0]/self.bins
                img_b = self.__load_img(subset_b[k])
                hist_b = hist_b + np.histogram(img_b.flatten(), bins = self.bins)[0]/self.bins
               
            # 4) Calculate dissimilarity
            d = np.abs((np.mean(hist_a)-np.mean(hist_b)))/(np.mean(hist_a))
            #print(d)

            # 5) Save split
            df2 = pd.DataFrame({'Iteration':i, 'Subset-A':[subset_a], 'Subset-B':[subset_b], 'Dissimilarity':d})
            df = df.append(df2)

        # Get the subsets with the minimum value of dissimilarity
        result = df[df['Dissimilarity'] == df['Dissimilarity'].min()]
        final_a = result['Subset-A'].values[0]
        final_b = result['Subset-B'].values[0]

        return df, final_a, final_b
