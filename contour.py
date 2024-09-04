import cv2
import numpy as np
import matplotlib.pyplot as plt

# Provide the path to the image you want to read
image_path = "/kaggle/input/breast-cancer-jpg-image-dataset-of-cbisddsm/k_CBIS-DDSM/jpg_img/Calc_Test_P_00038_LEFT_MLO_1-1.3.6.1.4.1.9590.100.1.2.29112199613143138535387754440942211739-1.3.6.1.4.1.9590.100.1.2.188613955710170417803011787532523988680"

# Read the grayscale image using imread function (no need for color conversion)
image_bw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resizing the image for compatibility
image_bw = cv2.resize(image_bw, (500, 600))

# Apply Gaussian filtering to reduce noise
image_blur = cv2.GaussianBlur(image_bw, (3, 3), 0)

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit=2)
final_img = clahe.apply(image_blur) + 30

# Thresholding to create a binary image
_, binary_image = cv2.threshold(final_img, 200, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image (coloring contours in green here)
contour_img = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_img, contours, -1, (220, 245 , 245), 2)

# Showing the CLAHE-enhanced image with contours using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_bw, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(contour_img)
plt.title("CLAHE Image with Contours")

plt.show()
