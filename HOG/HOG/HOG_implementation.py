import skimage.io, skimage.color
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
import HOG

img = skimage.io.imread("fotofamilia - Copy.bmp")
img = skimage.color.rgb2gray(img)

horizontal_mask = np.array([-1, 0, 1])
vertical_mask = np.array([[-1],
                             [0],
                             [1]])


print(len(img[0]))

horizontal_gradient = HOG.calculate_gradient(img, horizontal_mask)
vertical_gradient = HOG.calculate_gradient(img, vertical_mask)

# invVert = (skimage.color.rgb2gray((vertical_gradient)))
# invHor = skimage.util.invert(skimage.color.rgb2gray((horizontal_gradient)))


# invVert = exposure.rescale_intensity(invVert, in_range=(0.3,0.8)) 

# skimage.io.imshow(invVert,cmap=plt.cm.gray)
# skimage.io.imshow(vertical_gradient, cmap=plt.cm.gray)
# plt.show()
print(horizontal_gradient)
print(horizontal_gradient.shape)

# quit()

# skimage.io.imsave("invVert.png",invVert)




# skimage.io.imshow(invHor,cmap=plt.cm.gray)
# plt.show()

grad_magnitude = HOG.gradient_magnitude(horizontal_gradient, vertical_gradient)
grad_direction = HOG.gradient_direction(horizontal_gradient, vertical_gradient)
print(grad_direction[0][0])

# skimage.io.imshow(grad_magnitude)
# plt.show()
# skimage.io.imshow(grad_direction)
# plt.show()
# quit()

grad_direction = grad_direction % 180
hist_bins = np.array([10,30,50,70,90,110,130,150,170])

# Histogram of the first cell in the first block.
cell_direction = grad_direction[:8, :8]
cell_magnitude = grad_magnitude[:8, :8]
HOG_cell_hist = HOG.HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)

plt.bar(x=np.arange(9), height=HOG_cell_hist, align="center", width=0.8)
plt.show()

