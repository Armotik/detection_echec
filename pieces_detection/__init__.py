import cv2
import numpy as np

from numpy import ndarray


def is_cell_empty(cell: tuple[tuple[int, int], np.ndarray], threshold=0.05) -> bool:
    """
    Vérifie si une cellule est vide en fonction d'un seuil de pixels blancs.
    :param cell: La cellule à vérifier
    :param threshold: le seuil de pixels blancs pour considérer la cellule comme vide
    :return: True si la cellule est vide, False sinon.
    """
    _, img = cell
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    border = 5
    cropped = morph[border:-border, border:-border]


    white_pixels = np.sum(cropped == 255)
    total_pixels = cropped.size
    white_ratio = white_pixels / total_pixels

    return white_ratio < threshold
