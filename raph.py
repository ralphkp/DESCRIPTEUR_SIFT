import cv2
import numpy as np
from matplotlib import pyplot as plt

# Charger les images
img1 = cv2.imread('objets/thaprua.jpg', 0)  # Image de requête
img2 = cv2.imread('imagesRecherche/20221206_003418_small.jpg', 0) # Image de formation

# Initialiser SIFTD
sift = cv2.SIFT_create()

# Trouver les points clés et les descripteurs avec SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Paramètres FLANN et matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Filtrer les correspondances
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# Définir le nombre minimum de correspondances
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print(f"Pas assez de correspondances trouvées - {len(good)}/{MIN_MATCH_COUNT}")
    # Important de continuer même s'il n'y a pas assez de correspondances
    matchesMask = None

# Préparer matchesMask pour drawMatches comme attendu
if len(good) > MIN_MATCH_COUNT:
    matchesMask = mask.ravel().tolist()
else:
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,  # Appliquer le masque pour les inliers seulement
                   flags=2)

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)  # Conversion pour matplotlib

plt.imshow(img_matches)
plt.title('Caractéristiques correspondantes')
plt.show()

# Sauvegarder le résultat
output_path = 'test/thaprua_matches.jpg'
cv2.imwrite(output_path, img_matches)
print(f"Résultat sauvegardé dans {output_path}")