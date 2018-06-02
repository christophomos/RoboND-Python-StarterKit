import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

ground01_filename = 'images/ground01.jpg'
ground01_image = mpimg.imread(ground01_filename)
plt.imshow(ground01_image)
plt.show()

print("ground01_image.dtype",ground01_image.dtype)
print("ground01_image.shape",ground01_image.shape)
print("ground01_image.max()",ground01_image.max())
print("ground01_image.min()",ground01_image.min())

plt.clf()

red_channel = np.copy(ground01_image)
red_channel[:,:,[1,2]] = 0
green_channel = np.copy(ground01_image)
green_channel[:,:,[0,2]] = 0
blue_channel = np.copy(ground01_image)
blue_channel[:,:,[0,1]] = 0

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3))
ax1.imshow(red_channel)
ax2.imshow(green_channel)
ax3.imshow(blue_channel)
plt.show()

def lower_and_upper(image, channel, min_x, max_x, min_y, max_y, z_value = 6.0):
    plt.clf()
    cropped = np.array(image[min_y:max_y, min_x:max_x, channel])
    plt.imshow(cropped, cmap="gray")
    plt.show()
    cropped = cropped.flatten()
    cropped_mean = cropped.mean()
    cropped_std = cropped.std()
    distance_from_mean = cropped_std * z_value
    lower_bound = cropped_mean - distance_from_mean
    upper_bound = cropped_mean + distance_from_mean
    return (np.rint(lower_bound).astype(int), np.rint(upper_bound).astype(int))

r_bounds = lower_and_upper(ground01_image, 0, 0, 320, 130, 160)
print("r_bounds",r_bounds)
g_bounds = lower_and_upper(ground01_image, 1, 0, 320, 130, 160)
print("g_bounds",g_bounds)
b_bounds = lower_and_upper(ground01_image, 2, 0, 320, 130, 160)
print("b_bounds",b_bounds)

mask_lower = np.array([r_bounds[0], g_bounds[0], b_bounds[0]])
mask_upper = np.array([r_bounds[1], g_bounds[1], b_bounds[1]])

mask = cv2.inRange(ground01_image, mask_lower, mask_upper)
plt.clf()
plt.imshow(mask, cmap="gray")
plt.show()