import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def process_images(img1, kp1, des1, img2_path):
    img2 = cv2.imread(img2_path, 0)  # Charger l'image de formation

    # Trouver les points clés et les descripteurs de l'image de formation avec SIFT
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Trouver les correspondances
    matches = flann.knnMatch(des1, des2, k=2)

    # Filtrer les correspondances
    good = [m for m, n in matches if m.distance < 0.7*n.distance]

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Vérifier si une matrice de transformation valide a été trouvée
        if M is not None:
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("La matrice de transformation est None. Impossible de calculer perspectiveTransform.")
            matchesMask = None
    else:
        print(f"Pas assez de correspondances trouvées - {len(good)}/{MIN_MATCH_COUNT}")
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Appliquer le masque pour les inliers seulement
                       flags=2)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return img_matches

# Charger l'image de requête
img1 = cv2.imread('objets/thaprua.jpg', 0)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)

# Paramètres FLANN et matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

MIN_MATCH_COUNT = 20

# Chemin vers le répertoire contenant les images de formation
images_formation_directory = 'imagesRecherche'

# Boucler sur chaque fichier dans le répertoire des images de formation
for filename in os.listdir(images_formation_directory):
    filepath = os.path.join(images_formation_directory, filename)
    if filepath.endswith(".jpg"):  # Assurez-vous de traiter seulement les fichiers .jpg
        print(f"Traitement de {filename}")
        
        img_matches = process_images(img1, kp1, des1, filepath)
        
        # Sauvegarder les images résultantes
        output_path = os.path.join('data', f'matches_{filename}')
        cv2.imwrite(output_path, img_matches)
        print(f"Résultat sauvegardé dans {output_path}")
