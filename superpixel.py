#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[1]:


import cv2
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt

# Load the fundus image
image_path = 'E:\\fundus1.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply SLIC superpixel segmentation
num_segments = 100  # You can adjust the number of segments based on your image and preference
segments = slic(image_rgb, n_segments=num_segments, compactness=10, sigma=1)

# Create a mask for visualization
mask = np.zeros_like(image_rgb)
for label in np.unique(segments):
    mask[segments == label] = np.mean(image_rgb[segments == label], axis=0)

# Display the original image and superpixel segmentation result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.title('Superpixel Segmentation')
plt.axis('off')

plt.show()


# In[7]:


import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt

# Load the fundus image
image_path = 'E:\\fundus1.png'
image = cv2.imread(image_path)

# Superpixel Segmentation
segments = slic(image, n_segments=100, compactness=10, sigma=1)

# Find initial contour points based on superpixel segments
def find_initial_contour(segments):
    # Implement your logic to find initial contour points
    # For example, find the centroid of each superpixel segment
    contour_points = []
    for segment in segments:
        centroid_x = np.mean(segment[0])
        centroid_y = np.mean(segment[1])
        contour_points.append([centroid_x, centroid_y])
    return np.array(contour_points)

def map_contour_to_original_space(snake, segments):
    # Implement your logic to map contour points to original image space
    # For example, find the original coordinates of superpixel centroids
    original_contour_points = []
    for point in snake:
        segment_index = segments[int(point[0]), int(point[1])]
        segment_centroid = np.mean(np.argwhere(segments == segment_index), axis=0)
        original_contour_points.append(segment_centroid)
    return np.array(original_contour_points)


initial_contour = find_initial_contour(segments)

# Apply Active Contour on Superpixel Segments
snake = active_contour(find_boundaries(segments), initial_contour, alpha=0.01, beta=0.1, gamma=0.001)

# Map contour points from superpixel space to original image space
def map_contour_to_original_space(snake, segments):
    # For example, find the original coordinates of superpixel centroids
    original_contour_points = []
    for point in snake:
        segment_index = segments[int(point[0]), int(point[1])]
        segment_centroid = np.mean(np.argwhere(segments == segment_index), axis=0)
        original_contour_points.append(segment_centroid)
    return np.array(original_contour_points)

contour_points_original = map_contour_to_original_space(snake, segments)

# Visualize the results
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(mark_boundaries(image, segments, color=(0, 1, 0), mode='outer'))
ax.plot(contour_points_original[:, 1], contour_points_original[:, 0], '--r', linewidth=2, label='Active Contour')
ax.legend()
plt.axis('off')
plt.show()


# In[2]:


import cv2
import numpy as np
from skimage.segmentation import slic, find_boundaries, active_contour
from skimage import exposure
import matplotlib.pyplot as plt

# Load the fundus image
image_path = 'E:\\fundus1.png'
image = cv2.imread(image_path)

# Enhance contrast using histogram equalization
image_equalized = exposure.equalize_hist(image)

# Superpixel Segmentation
segments = slic(image_equalized, n_segments=100, compactness=10, sigma=1)

# Find initial contour points based on the top three brightest superpixel segments
def find_initial_contour(segments, image):
    # Get average intensity of each superpixel
    segment_intensities = [np.mean(image[segments == i]) for i in range(segments.max() + 1)]

    # Remove NaN values from the list
    segment_intensities = [value for value in segment_intensities if not np.isnan(value)]

    # Select top three brightest segments
    brightest_segments = np.argsort(segment_intensities)[-3:]

    # Find initial contour points based on the centroids of the selected segments
    contour_points = []
    for segment_index in brightest_segments:
        centroid = np.mean(np.argwhere(segments == segment_index), axis=0)
        contour_points.append(centroid)
    return np.array(contour_points)


initial_contour = find_initial_contour(segments, image)

# Apply Active Contour on Superpixel Segments
snake = active_contour(find_boundaries(segments), initial_contour, alpha=0.01, beta=0.1, gamma=0.001)

# Map contour points from superpixel space to original image space
def map_contour_to_original_space(snake, segments):
    original_contour_points = []
    for point in snake:
        segment_index = segments[int(point[0]), int(point[1])]
        segment_centroid = np.mean(np.argwhere(segments == segment_index), axis=0)
        original_contour_points.append(segment_centroid)
    return np.array(original_contour_points)

contour_points_original = map_contour_to_original_space(snake, segments)

# Visualize the results
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_equalized)
ax.plot(contour_points_original[:, 1], contour_points_original[:, 0], '--r', linewidth=2, label='Active Contour')
ax.legend()
plt.axis('off')
plt.show()


# In[39]:


import os
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import active_contour, find_boundaries

def resize_image(image, target_size):
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    return image



def check_shape_equality(*images):
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')

def hybrid_active_contour_superpixel(image_path, target_size=(256, 256), n_segments=100, compactness=10, sigma=1):
    # Load the image
    original_image = cv2.imread(image_path)
    
    # Resize the image
    original_image_resized = resize_image(original_image, target_size)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2GRAY)
    
    # Superpixel Segmentation using SLIC
    segments = slic(original_image_resized, n_segments=n_segments, compactness=compactness, sigma=sigma)
    
    # Resize the segments to match the resized image
    resized_segments = resize_image(segments, target_size).astype(int)
    
    # Initialize contour points based on superpixel segments
    initial_contour = find_initial_contour(resized_segments)
    
    # Apply Active Contour on Superpixel Segments
    snake = active_contour(find_boundaries(resized_segments), initial_contour, alpha=0.01, beta=0.1, gamma=0.001)
    
    # Map contour points from superpixel space to original image space
    contour_points_original = map_contour_to_original_space(snake, resized_segments)
    
    # Visualize the results
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(original_image_resized, resized_segments, color=(0, 1, 0), mode='outer'))
    plt.plot(contour_points_original[:, 1], contour_points_original[:, 0], '--r', linewidth=2, label='Active Contour')
    plt.legend()
    plt.axis('off')
    plt.show()
    
    return original_image_resized, resized_segments


def find_initial_contour(segments):
    # Implement your logic to find initial contour points
    # For example, find the centroid of each superpixel segment
    contour_points = []
    for segment in segments:
        centroid_x = np.mean(segment[0])
        centroid_y = np.mean(segment[1])
        contour_points.append([centroid_x, centroid_y])
    return np.array(contour_points)

def map_contour_to_original_space(snake, segments):
    # Implement your logic to map contour points to original image space
    # For example, find the original coordinates of superpixel centroids
    original_contour_points = []
    for point in snake:
        i, j = int(point[0]), int(point[1])
        if 0 <= i < segments.shape[0] and 0 <= j < segments.shape[1]:
            segment_index = segments[i, j]
            segment_centroid = np.mean(np.argwhere(segments == segment_index), axis=0)
            original_contour_points.append(segment_centroid)
    return np.array(original_contour_points)

def evaluate_segmentation(original_image, segments):
    # Calculate silhouette score
    silhouette = silhouette_score(original_image.reshape(-1, 3), segments.flatten())

    # Calculate Calinski Harabasz index
    calinski_harabasz = calinski_harabasz_score(original_image.reshape(-1, 3), segments.flatten())

    return silhouette, calinski_harabasz

def check_shape_equality(*images):
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')

def calculate_psnr(original_image, segmented_image):
    # Resize the original image to match the dimensions of the segmented image
    original_image_resized = resize_image(original_image, segmented_image.shape[:2])

    # Ensure both images have the same dimensions
    check_shape_equality(original_image_resized, segmented_image)

    # Convert images to uint8
    original_image_uint8 = img_as_ubyte(original_image_resized)
    segmented_image_uint8 = img_as_ubyte(segmented_image)

    # Calculate PSNR
    return peak_signal_noise_ratio(original_image_uint8, segmented_image_uint8)
# Directory containing images
dataset_dir = 'E://erm1'

# List to store evaluation results
silhouette_scores = []
calinski_harabasz_scores = []
psnr_values = []

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.lower().endswith('.jpg'):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_name)
        
        # Apply Hybrid Active Contour with Superpixelation
        original_image, segments = hybrid_active_contour_superpixel(image_path)
        
        # Evaluate the segmentation
        silhouette, calinski_harabasz = evaluate_segmentation(original_image, segments)
        psnr = calculate_psnr(original_image, segments)
        
        # Print or store the evaluation results
        print(f"Image: {image_name}, Silhouette Score: {silhouette}, Calinski-Harabasz Index: {calinski_harabasz}, PSNR: {psnr}")
        
        # Store the results for further analysis if needed
        silhouette_scores.append(silhouette)
        calinski_harabasz_scores.append(calinski_harabasz)
        psnr_values.append(psnr)
    else:
        print(f"Skipping {image_name} as it does not have '.jpg' file format.")

# Calculate and print average silhouette score, Calinski-Harabasz index, and PSNR
average_silhouette = np.mean(silhouette_scores)
average_calinski_harabasz = np.mean(calinski_harabasz_scores)
average_psnr = np.mean(psnr_values)
print(f"\nAverage Silhouette Score: {average_silhouette}, Average Calinski-Harabasz Index: {average_calinski_harabasz}, Average PSNR: {average_psnr}")


# In[43]:


import os
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.segmentation import active_contour

def resize_image(image, target_size):
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    return image

def hybrid_active_contour_superpixel(image_path, target_size=(256, 256), n_segments=100, compactness=10, sigma=1):
    # Load the image
    original_image = cv2.imread(image_path)
    
    # Resize the image
    original_image_resized = resize_image(original_image, target_size)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2GRAY)
    
    # Superpixel Segmentation using SLIC
    segments = slic(original_image_resized, n_segments=n_segments, compactness=compactness, sigma=sigma)
    
    # Initialize contour points based on superpixel segments
    initial_contour = find_initial_contour(segments)
    
    # Apply Active Contour on Superpixel Segments
    snake = active_contour(find_boundaries(segments), initial_contour, alpha=0.01, beta=0.1, gamma=0.001)
    
    # Map contour points from superpixel space to original image space
    contour_points_original = map_contour_to_original_space(snake, segments)
    
    # Visualize the results
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(original_image_resized, segments, color=(0, 1, 0), mode='outer'))
    plt.plot(contour_points_original[:, 1], contour_points_original[:, 0], '--r', linewidth=2, label='Active Contour')
    plt.legend()
    plt.axis('off')
    plt.show()
    
    return original_image_resized, segments

def find_initial_contour(segments):
    # Implement your logic to find initial contour points
    # For example, find the centroid of each superpixel segment
    contour_points = []
    for segment in segments:
        centroid_x = np.mean(segment[0])
        centroid_y = np.mean(segment[1])
        contour_points.append([centroid_x, centroid_y])
    return np.array(contour_points)

def map_contour_to_original_space(snake, segments):
    # Implement your logic to map contour points to original image space
    # For example, find the original coordinates of superpixel centroids
    original_contour_points = []
    for point in snake:
        segment_index = segments[int(point[0]), int(point[1])]
        segment_centroid = np.mean(np.argwhere(segments == segment_index), axis=0)
        original_contour_points.append(segment_centroid)
    return np.array(original_contour_points)

def evaluate_segmentation(original_image, segments):
    # Calculate silhouette score
    silhouette = silhouette_score(original_image.reshape(-1, 3), segments.flatten())

    # Calculate Calinski Harabasz index
    calinski_harabasz = calinski_harabasz_score(original_image.reshape(-1, 3), segments.flatten())

    return silhouette, calinski_harabasz

def calculate_psnr(original_image, segmented_image):
    # Resize both images to have the same dimensions
    original_image_resized = resize_image(original_image, segmented_image.shape[:2])

    # Ensure both images have the same dimensions
    check_shape_equality(original_image_resized, segmented_image)

    # Convert images to uint8
    original_image_uint8 = img_as_ubyte(original_image_resized)
    segmented_image_uint8 = img_as_ubyte(segmented_image)

    # Calculate PSNR
    return peak_signal_noise_ratio(original_image_uint8, segmented_image_uint8)


def check_shape_equality(*images):
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')

# Directory containing images
dataset_dir = 'E:/erm1'

# List to store evaluation results
silhouette_scores = []
calinski_harabasz_scores = []
psnr_scores = []

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.lower().endswith('.jpg'):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_name)
        
        # Apply Hybrid Active Contour with Superpixelation
        original_image, segments = hybrid_active_contour_superpixel(image_path)
        
        # Evaluate the segmentation
        silhouette, calinski_harabasz = evaluate_segmentation(original_image, segments)
        
        # Calculate PSNR
        psnr = calculate_psnr(original_image, segments)
        
        # Print or store the evaluation results
        print(f"Image: {image_name}, Silhouette Score: {silhouette}, Calinski-Harabasz Index: {calinski_harabasz}, PSNR: {psnr}")
        
        # Store the results for further analysis if needed
        silhouette_scores.append(silhouette)
        calinski_harabasz_scores.append(calinski_harabasz)
        psnr_scores.append(psnr)
    else:
        print(f"Skipping {image_name} as it does not have '.jpg' file format.")

# Calculate and print average silhouette score, Calinski-Harabasz index, and PSNR
average_silhouette = np.mean(silhouette_scores)
average_calinski_harabasz = np.mean(calinski_harabasz_scores)
average_psnr = np.mean(psnr_scores)

print(f"\nAverage Silhouette Score: {average_silhouette}")
print(f"Average Calinski-Harabasz Index: {average_calinski_harabasz}")
print(f"Average PSNR: {average_psnr}")


# In[47]:


#superpixelation
import os
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def apply_superpixelation(image_path, target_size=(256, 256), n_segments=100, compactness=10, sigma=1):
    # Load the image
    original_image = cv2.imread(image_path)
    
    # Resize the image
    original_image_resized = resize_image(original_image, target_size)
    
    # Superpixel Segmentation using SLIC
    segments = slic(original_image_resized, n_segments=n_segments, compactness=compactness, sigma=sigma)
    
    # Visualize the superpixelation
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(original_image_resized, segments, color=(0, 1, 0), mode='outer'))
    plt.axis('off')
    plt.show()
    
    return original_image_resized, segments

def evaluate_superpixelation(original_image, segments):
    # Calculate silhouette score
    silhouette = silhouette_score(original_image.reshape(-1, 3), segments.flatten())

    # Calculate Calinski Harabasz index
    calinski_harabasz = calinski_harabasz_score(original_image.reshape(-1, 3), segments.flatten())

    return silhouette, calinski_harabasz

def check_shape_equality(*images):
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')

# Directory containing fundus images
dataset_dir = 'E:/erm1'

# List to store evaluation results
silhouette_scores = []
calinski_harabasz_scores = []
psnr_scores = []

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.lower().endswith('.jpg'):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_name)
        
        # Apply Superpixelation
        original_image, segments = apply_superpixelation(image_path)
        
        # Evaluate the Superpixelation
        silhouette, calinski_harabasz = evaluate_superpixelation(original_image, segments)
        
        # Print or store the evaluation results
        print(f"Image: {image_name}, Silhouette Score: {silhouette}, Calinski-Harabasz Index: {calinski_harabasz}")
        
        # Store the results for further analysis if needed
        silhouette_scores.append(silhouette)
        calinski_harabasz_scores.append(calinski_harabasz)
    else:
        print(f"Skipping {image_name} as it does not have '.jpg' file format.")

# Calculate and print average silhouette score, Calinski-Harabasz index, and PSNR
average_silhouette = np.mean(silhouette_scores)
average_calinski_harabasz = np.mean(calinski_harabasz_scores)
print(f"\nAverage Silhouette Score: {average_silhouette}, Average Calinski-Harabasz Index: {average_calinski_harabasz}")


# In[48]:


#hybrid active contour model
import os
import cv2
import numpy as np
from skimage.segmentation import active_contour, slic
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def apply_active_contour(image, alpha=0.01, beta=10, gamma=0.01, iterations=250, sigma=3):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the snake (active contour)
    snake = active_contour(gray_image, alpha * np.array([1, 1]), beta=beta, gamma=gamma, max_iterations=iterations, sigma=sigma)
    
    return snake

def apply_kmeans_clustering(image, n_clusters=3):
    # Flatten the image to a 1D array
    flat_image = image.reshape(-1, 3)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(flat_image)
    
    # Reshape the labels to the original image shape
    segmented_image = labels.reshape(image.shape[:2])
    
    return segmented_image

def evaluate_segmentation(original_image, segmented_image):
    # Flatten the original image and the segmented image
    flat_original = original_image.reshape(-1, 3)
    flat_segmented = segmented_image.flatten()

    # Calculate silhouette score
    silhouette = silhouette_score(flat_original, flat_segmented)

    # Calculate Calinski Harabasz index
    calinski_harabasz = calinski_harabasz_score(flat_original, flat_segmented)

    # Calculate Davies-Bouldin index
    davies_bouldin = davies_bouldin_score(flat_original, flat_segmented)

    return silhouette, calinski_harabasz, davies_bouldin

# Directory containing images
dataset_dir = 'E:/erm1'

# List to store evaluation results
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.lower().endswith('.jpg'):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_name)
        
        # Load the image
        original_image = cv2.imread(image_path)
        
        # Resize the image
        original_image_resized = resize_image(original_image, (256, 256))
        
        # Apply K-Means clustering for unsupervised segmentation
        segmented_image = apply_kmeans_clustering(original_image_resized)
        
        # Evaluate the segmentation
        silhouette, calinski_harabasz, davies_bouldin = evaluate_segmentation(original_image_resized, segmented_image)
        
        # Print or store the evaluation results
        print(f"Image: {image_name}, Silhouette Score: {silhouette}, Calinski-Harabasz Index: {calinski_harabasz}, Davies-Bouldin Index: {davies_bouldin}")
        
        # Store the results for further analysis if needed
        silhouette_scores.append(silhouette)
        calinski_harabasz_scores.append(calinski_harabasz)
        davies_bouldin_scores.append(davies_bouldin)
    else:
        print(f"Skipping {image_name} as it does not have '.jpg' file format.")

# Calculate and print average silhouette score, Calinski-Harabasz index, and Davies-Bouldin index
average_silhouette = np.mean(silhouette_scores)
average_calinski_harabasz = np.mean(calinski_harabasz_scores)
average_davies_bouldin = np.mean(davies_bouldin_scores)
print(f"\nAverage Silhouette Score: {average_silhouette}, Average Calinski-Harabasz Index: {average_calinski_harabasz}, Average Davies-Bouldin Index: {average_davies_bouldin}")


# In[49]:


import os
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def hybrid_active_contour_superpixel(image_path, target_size=(256, 256), n_segments=100, compactness=10, sigma=1):
    # Load the image
    original_image = cv2.imread(image_path)
    
    # Resize the image
    original_image_resized = resize_image(original_image, target_size)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2GRAY)
    
    # Superpixel Segmentation using SLIC
    segments = slic(original_image_resized, n_segments=n_segments, compactness=compactness, sigma=sigma)
    
    # Initialize contour points based on superpixel segments
    initial_contour = find_initial_contour(segments)
    
    # Apply Active Contour on Superpixel Segments
    snake = active_contour(find_boundaries(segments), initial_contour, alpha=0.01, beta=0.1, gamma=0.001)
    
    # Map contour points from superpixel space to original image space
    contour_points_original = map_contour_to_original_space(snake, segments)
    
    # Visualize the results
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(original_image_resized, segments, color=(0, 1, 0), mode='outer'))
    plt.plot(contour_points_original[:, 1], contour_points_original[:, 0], '--r', linewidth=2, label='Active Contour')
    plt.legend()
    plt.axis('off')
    plt.show()
    
    return original_image_resized, segments

def find_initial_contour(segments):
    # Implement your logic to find initial contour points
    # For example, find the centroid of each superpixel segment
    contour_points = []
    for segment in segments:
        centroid_x = np.mean(segment[0])
        centroid_y = np.mean(segment[1])
        contour_points.append([centroid_x, centroid_y])
    return np.array(contour_points)

def map_contour_to_original_space(snake, segments):
    # Implement your logic to map contour points to original image space
    # For example, find the original coordinates of superpixel centroids
    original_contour_points = []
    for point in snake:
        segment_index = segments[int(point[0]), int(point[1])]
        segment_centroid = np.mean(np.argwhere(segments == segment_index), axis=0)
        original_contour_points.append(segment_centroid)
    return np.array(original_contour_points)

def evaluate_segmentation(original_image, segments):
    # Calculate silhouette score
    silhouette = silhouette_score(original_image.reshape(-1, 3), segments.flatten())

    # Calculate Calinski Harabasz index
    calinski_harabasz = calinski_harabasz_score(original_image.reshape(-1, 3), segments.flatten())

    # Calculate Davies-Bouldin index
    davies_bouldin = davies_bouldin_score(original_image.reshape(-1, 3), segments.flatten())

    return silhouette, calinski_harabasz, davies_bouldin

# Directory containing images
dataset_dir = 'E:/erm1'

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.lower().endswith('.jpg'):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_name)
        
        # Apply Hybrid Active Contour with Superpixelation
        original_image, segments = hybrid_active_contour_superpixel(image_path)
        
        # Evaluate the segmentation
        silhouette, calinski_harabasz, davies_bouldin = evaluate_segmentation(original_image, segments)
        
        # Print or store the evaluation results
        print(f"Image: {image_name}, Silhouette Score: {silhouette}, Calinski-Harabasz Index: {calinski_harabasz}, Davies-Bouldin Index: {davies_bouldin}")
        
    else:
        print(f"Skipping {image_name} as it does not have '.jpg' file format.")


# In[ ]:




