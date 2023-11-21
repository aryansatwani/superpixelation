#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fundus image
image_path = 'E:\\fundus1.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(gray_image)

# Define a Gaussian-shaped filter (adjust sigma based on vessel width)
sigma = 1.0
filter_size = int(6 * sigma)
if filter_size % 2 == 0:
    filter_size += 1

gaussian_filter = cv2.getGaussianKernel(filter_size, sigma)
gaussian_filter = gaussian_filter * gaussian_filter.T

# Perform matched filtering
filtered_image = cv2.filter2D(enhanced_image, -1, gaussian_filter)

# Adaptive Thresholding
binary_segmentation = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Display the original image, enhanced image, filtered image, and segmentation result
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(binary_segmentation, cmap='gray')
plt.title('Segmentation Result')
plt.axis('off')

plt.show()



# In[10]:


#tried with otsu's thresholding method and it was not great
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fundus image
image_path = 'E:\\fundus1.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(gray_image)

# Apply median blur to reduce noise
blurred_image = cv2.medianBlur(enhanced_image, 5)

# Define a Gaussian-shaped filter (adjust sigma based on vessel width)
sigma = 1.0
filter_size = int(6 * sigma)
if filter_size % 2 == 0:
    filter_size += 1

gaussian_filter = cv2.getGaussianKernel(filter_size, sigma)
gaussian_filter = gaussian_filter * gaussian_filter.T

# Perform matched filtering
filtered_image = cv2.filter2D(blurred_image, -1, gaussian_filter)

# Otsu's Thresholding
_, binary_segmentation = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original image, enhanced image, filtered image, and segmentation result
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(binary_segmentation, cmap='gray')
plt.title('Segmentation Result (Otsu\'s Thresholding)')
plt.axis('off')

plt.show()


# In[12]:


import cv2
import numpy as np
import os

# Function to preprocess an image
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)
    return enhanced_image

# Function to segment blood vessels using matched filtering
def segment_blood_vessels(image):
    sigma = 1.0
    filter_size = int(6 * sigma)
    if filter_size % 2 == 0:
        filter_size += 1
    gaussian_filter = cv2.getGaussianKernel(filter_size, sigma)
    gaussian_filter = gaussian_filter * gaussian_filter.T
    filtered_image = cv2.filter2D(image, -1, gaussian_filter)
    binary_segmentation = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_segmentation

# Path to the folder containing fundus images
folder_path = 'E:\\fundus_imgs'

# List to store segmentation results
segmentation_results = []

# Loop through the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Segment blood vessels
        segmented_image = segment_blood_vessels(preprocessed_image)

        # Store the segmentation result
        segmentation_results.append(segmented_image)

# Now segmentation_results list contains binary segmentation masks for all images in the folder.
# You can further process these masks or use them for evaluation.


# In[13]:


import cv2
import os
import numpy as np

# Function to calculate Intersection over Union (IoU)
def calculate_iou(gt_mask, predicted_mask):
    intersection = np.logical_and(gt_mask, predicted_mask)
    union = np.logical_or(gt_mask, predicted_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Path to the folder containing ground truth masks
gt_folder_path = 'E:\\fundus_imgs'

# List to store IoU scores
iou_scores = []

# Loop through the ground truth masks and corresponding predicted masks
for gt_filename in os.listdir(gt_folder_path):
    if gt_filename.endswith('.png'):
        # Load ground truth mask
        gt_mask_path = os.path.join(gt_folder_path, gt_filename)
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

        # Find corresponding predicted mask from segmentation_results list
        # (assuming the filenames match between ground truth and predictions)
        predicted_mask = segmentation_results[index]  # Replace 'index' with the appropriate index

        # Calculate IoU score
        iou_score = calculate_iou(gt_mask > 0, predicted_mask > 0)
        iou_scores.append(iou_score)

# Calculate average IoU score
average_iou = np.mean(iou_scores)
print(f'Average IoU Score: {average_iou}')


# In[14]:


gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)


# In[25]:


import cv2
import os
import numpy as np
from skimage.segmentation import active_contour
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans

# Directory containing fundus images
dataset_dir = 'E:\\chase'

# Lists to store evaluation metrics for all valid images
silhouette_scores = []
calinski_harabasz_scores = []

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.endswith('.jpg'):
        # Load the fundus image
        image_path = os.path.join(dataset_dir, image_name)
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            continue

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)

        # Define a Gaussian-shaped filter (adjust sigma based on vessel width)
        sigma = 1.0
        filter_size = int(6 * sigma)
        if filter_size % 2 == 0:
            filter_size += 1

        gaussian_filter = cv2.getGaussianKernel(filter_size, sigma)
        gaussian_filter = gaussian_filter * gaussian_filter.T

        # Perform matched filtering
        filtered_image = cv2.filter2D(enhanced_image, -1, gaussian_filter)

        # Adaptive Thresholding
        binary_segmentation = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Ensure binary image has non-zero size
        if np.count_nonzero(binary_segmentation) == 0:
            print(f"Error: Binary image has zero size for image {image_path}")
            continue

        # Apply clustering (e.g., KMeans) with different parameters
        flat_image = binary_segmentation.flatten()
        try:
            # Apply clustering (e.g., KMeans) with different parameters
            labels = KMeans(n_clusters=2, random_state=42).fit_predict(flat_image.reshape(-1, 1))

            # Print unique labels obtained from clustering
            print(f"Unique labels: {np.unique(labels)}")

            # Silhouette Score
            silhouette = silhouette_score(flat_image.reshape(-1, 1), labels)
            silhouette_scores.append(silhouette)

            # Calinski-Harabasz Index
            calinski_harabasz = calinski_harabasz_score(flat_image.reshape(-1, 1), labels)
            calinski_harabasz_scores.append(calinski_harabasz)

        except Exception as e:
            print(f"Error during clustering for image {image_path}: {e}")

# Calculate average scores
average_silhouette = np.nanmean(silhouette_scores)
average_calinski_harabasz = np.nanmean(calinski_harabasz_scores)

# Print average scores
print(f"Average Silhouette Score: {average_silhouette}")
print(f"Average Calinski-Harabasz Index: {average_calinski_harabasz}")



# In[50]:


import os
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def improved_matched_filtering(image_path, target_size=(256, 256)):
    # Load the image
    original_image = cv2.imread(image_path)
    
    # Resize the image
    original_image_resized = resize_image(original_image, target_size)
    
    # Convert the image to grayscale and handle floating-point values
    gray_image = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2GRAY)
    gray_image = img_as_ubyte(gray_image)
    
    # Define your filter (you may need to adjust the kernel based on your specific requirements)
    kernel = np.array([[1, 1, 1],
                       [1,  9, 1],
                       [1, 1, 1]])

    # Apply the improved matched filtering
    filtered_image = cv2.filter2D(gray_image, -1, kernel)

    return original_image_resized, filtered_image

def evaluate_segmentation(original_image, filtered_image):
    # Ensure both images have the same dimensions
    min_height, min_width = min(original_image.shape[0], filtered_image.shape[0]), min(original_image.shape[1], filtered_image.shape[1])
    original_image = original_image[:min_height, :min_width]
    filtered_image = filtered_image[:min_height, :min_width]

    # Calculate PSNR
    psnr = peak_signal_noise_ratio(original_image, filtered_image)

    return psnr

# Directory containing images
dataset_dir = 'E://erm1'

# List to store evaluation results
evaluation_metrics = []

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    if image_name.lower().endswith('.jpg'):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_name)
        
        # Apply Improved Matched Filtering
        original_image, filtered_image = improved_matched_filtering(image_path)
        
        # Evaluate the segmentation
        psnr = evaluate_segmentation(original_image, filtered_image)
        
        # Print or store the evaluation results
        print(f"Image: {image_name}, PSNR: {psnr}")
        
        # Store the results for further analysis if needed
        evaluation_metrics.append(psnr)
    else:
        print(f"Skipping {image_name} as it does not have '.jpg' file format.")

# Calculate and print average PSNR
average_psnr = np.mean(evaluation_metrics)
print(f"\nAverage PSNR: {average_psnr}")



# In[54]:





# In[55]:


import os
import cv2
import numpy as np
from skimage.segmentation import active_contour, mark_boundaries
from skimage.segmentation import slic
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def improved_match_filter(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Fourier Transform
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Create a mask to filter specific frequencies
    rows, cols = gray_image.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the circle in the frequency domain to keep
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    
    # Apply the mask to the frequency domain
    f_shift = f_shift * mask
    
    # Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the result to 0-255
    filtered_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return filtered_image.astype(np.uint8)



def hybrid_active_contour_match_filter(image_path, target_size=(256, 256), n_segments=100, compactness=10, sigma=1):
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
    
    # Apply Improved Match Filtering (you need to implement this)
    filtered_image = improved_match_filter(original_image_resized)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(mark_boundaries(original_image_resized, segments, color=(0, 1, 0), mode='outer'))
    plt.plot(contour_points_original[:, 1], contour_points_original[:, 0], '--r', linewidth=2, label='Active Contour')
    plt.title('Hybrid Active Contour Model')
    plt.legend()
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Improved Match Filtering Model')
    plt.axis('off')
    
    plt.show()
    
    return original_image_resized, segments, filtered_image

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

def improved_match_filter(image):
    # Implement your improved match filtering logic here
    # You might consider using Fourier Transform or other frequency domain techniques
    # Return the filtered image
    return image

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
        
        # Apply Hybrid Active Contour with Improved Match Filtering
        original_image, segments, filtered_image = hybrid_active_contour_match_filter(image_path)
        
        # Evaluate the segmentation
        silhouette, calinski_harabasz, davies_bouldin = evaluate_segmentation(original_image, segments)
        
        # Print or store the evaluation results
        print(f"Image: {image_name}, Silhouette Score: {silhouette}, Calinski-Harabasz Index: {calinski_harabasz}, Davies-Bouldin Index: {davies_bouldin}")
        
    else:
        print(f"Skipping {image_name} as it does not have '.jpg' file format.")

        


# In[60]:


import os
import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def improved_match_filter(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Fourier Transform
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Create a mask to filter specific frequencies
    rows, cols = gray_image.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the circle in the frequency domain to keep
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    
    # Apply the mask to the frequency domain
    f_shift = f_shift * mask
    
    # Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the result to 0-255
    filtered_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return filtered_image.astype(np.uint8)

def evaluate_segmentation(image, segments):
    # Ensure both image and segments have the same number of samples
    if image.size != segments.size:
        raise ValueError("Input image and segments must have the same number of samples.")
    
    # Calculate silhouette score
    silhouette = silhouette_score(image.reshape(-1, 3), segments.flatten())
    
    # Calculate Calinski Harabasz index
    calinski_harabasz = calinski_harabasz_score(image.reshape(-1, 3), segments.flatten())
    
    # Calculate Davies-Bouldin index
    davies_bouldin = davies_bouldin_score(image.reshape(-1, 3), segments.flatten())

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
        
        # Read the image
        original_image = cv2.imread(image_path)
        
        # Apply improved matched filtering
        filtered_image = improved_match_filter(original_image)
        
        # Convert single-channel image back to three channels
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
        
        # Superpixel Segmentation using SLIC
        segments = slic(filtered_image, n_segments=100, compactness=10, sigma=1, channel_axis=None)
        
        # Evaluate the segmentation
        silhouette, calinski_harabasz, davies_bouldin = evaluate_segmentation(filtered_image, segments)
        
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

print(f"\nAverage Silhouette Score: {average_silhouette}")
print(f"Average Calinski-Harabasz Index: {average_calinski_harabasz}")
print(f"Average Davies-Bouldin Index: {average_davies_bouldin}")



# In[66]:


import os
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from skimage.transform import resize
from tqdm import tqdm

def improved_matched_filtering(image_path, target_size=(256, 256)):
    # Load the image
    original_image = cv2.imread(image_path)
    
    # Resize the image to the target size
    original_image_resized = cv2.resize(original_image, target_size)

    # Convert the image to grayscale and handle floating-point values
    gray_image = cv2.cvtColor(original_image_resized, cv2.COLOR_BGR2GRAY)
    gray_image = img_as_ubyte(gray_image)

    # Define your filter (you may need to adjust the kernel based on your specific requirements)
    kernel = np.array([[1, 1, 1],
                       [1,  9, 1],
                       [1, 1, 1]])

    # Apply the filter to the image
    filtered_image = cv2.filter2D(gray_image, -1, kernel)

    return original_image_resized, filtered_image

def evaluate_segmentation(original_image, segments, filtered_image):
    # Convert segments to a color representation
    segmented_image = mark_boundaries(original_image, segments, color=(0, 1, 0), mode='outer')

    # Calculate silhouette score
    silhouette = silhouette_score(original_image.reshape(-1, 3), segments.flatten())

    # Calculate Calinski Harabasz index
    calinski_harabasz = calinski_harabasz_score(original_image.reshape(-1, 3), segments.flatten())

    # Resize the filtered image to match the original image dimensions
    filtered_image_resized = resize(filtered_image, original_image.shape)

    # Convert images to uint8 for metrics calculation
    original_image_uint8 = img_as_ubyte(original_image)
    filtered_image_uint8 = img_as_ubyte(filtered_image_resized)

    # Calculate metrics
    davies_bouldin = davies_bouldin_score(original_image_uint8.reshape(-1, 3), segments.flatten())

    return silhouette, calinski_harabasz, davies_bouldin, segmented_image

# Directory containing the dataset
dataset_dir = 'E:/erm1'

# Lists to store evaluation metrics for all images
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_values = []
segmented_images = []

# Loop through each image in the dataset
for image_name in os.listdir(dataset_dir):
    # Check if the file is a JPG image
    if image_name.lower().endswith('.jpg'):
        # Construct the full path to the image
        image_path = os.path.join(dataset_dir, image_name)

        # Apply the segmentation and get evaluation metrics and segmented image
        original_image, filtered_image = improved_matched_filtering(image_path)
        
        # Ensure the images are three-dimensional
        original_image = np.expand_dims(original_image, axis=-1) if original_image.ndim == 2 else original_image
        filtered_image = np.expand_dims(filtered_image, axis=-1) if filtered_image.ndim == 2 else filtered_image

        segments = slic(filtered_image, n_segments=100, compactness=10, sigma=1)
        silhouette, calinski_harabasz, davies_bouldin, segmented_image = evaluate_segmentation(original_image, segments, filtered_image)

        # Append the metrics and segmented image to the lists
        silhouette_scores.append(silhouette)
        calinski_harabasz_scores.append(calinski_harabasz)
        davies_bouldin_values.append(davies_bouldin)
        segmented_images.append(segmented_image)

# Calculate average metrics
average_silhouette = np.nanmean(silhouette_scores)
average_calinski_harabasz = np.nanmean(calinski_harabasz_scores)
average_davies_bouldin = np.nanmean(davies_bouldin_values)

# Print or store the average metrics
print(f"\nAverage Metrics:")
print(f"Silhouette Score: {average_silhouette}, Calinski-Harabasz Index: {average_calinski_harabasz}, Average Davies-Bouldin Index: {average_davies_bouldin}")


# In[ ]:




