import cv2
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy import ndimage

# Resmi yükle
image_path = "./Images/sporistanbulcrop4.png"
I = cv2.imread(image_path)

# Resmi RGB'ye dönüştürme
I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

# Resmi gri tona çevir
Igr = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Resme medyan fitresi uygulama
Imed = cv2.medianBlur(Igr,3)

# Otsu thresholding
T = threshold_otsu(Imed)
Ibw1 = Igr > T

# Ibw matrisini uint8 türüne dönüştür
Ibw1 = (Ibw1 * 255).astype(np.uint8)

# Morfolojik işlemler
T = threshold_otsu(Imed)
Ibw2 = Igr > T
Ibw2 = (Ibw2 * 255).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
Ibw2 = cv2.morphologyEx(Ibw2, cv2.MORPH_CLOSE, kernel)

# Sonucu göstermek için resimleri kaydedelim
cv2.imwrite('./Outputs/RGB.png',    I_rgb )
cv2.imwrite('./Outputs/Gray.png',   Igr)
cv2.imwrite('./Outputs/Median_Filter.png', Imed)
cv2.imwrite('./Outputs/Threasholding.png', Ibw1)
cv2.imwrite('./Outputs/Closing.png', Ibw2)


# İşlenen resimleri göster
plt.figure(figsize=(12, 8))
plt.subplot(321), plt.imshow(I_rgb,  cmap='gray'), plt.title('Rgb')
plt.subplot(322), plt.imshow(Igr,  cmap='gray'), plt.title('Gray')
plt.subplot(323), plt.imshow(Imed,  cmap='gray'), plt.title('Median Filter(Kernel 3x3) with Gray')
plt.subplot(324), plt.imshow(Ibw1,  cmap='gray'), plt.title('Adaptif Thresholding with 3x3 Median Filter')
plt.subplot(325), plt.imshow(Ibw2,  cmap='gray'), plt.title('Closing with Adaptif Thresholding')
plt.show()