import cv2
import numpy as np
from numpy import ndarray


def is_cell_white(cell: tuple[tuple[int, int], np.ndarray]) -> bool:
    """
    Vérifie si une cellule est blanche.
    :param cell: La cellule à vérifier
    :return: True si la cellule est blanche, False sinon.
    """
    _, img = cell
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuillage pour obtenir une image binaire
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Compter le nombre de pixels blancs
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size

    # Calculer le ratio de pixels blancs
    white_ratio = white_pixels / total_pixels

    return white_ratio < 0.5



def cell_compare(reference_cell: np.ndarray, current_cell: np.ndarray, is_white: bool, debug=True) -> bool:
    """
    Compare deux cellules par histogramme (centré, flouté) avec seuils adaptatifs selon couleur de case.

    :param reference_cell: Cellule vide de référence
    :param current_cell: Cellule actuelle
    :param is_white: True si case blanche, False si noire
    :param debug: Affiche la similarité
    :return: True si la cellule est considérée comme vide
    """



    if reference_cell.shape != current_cell.shape:
        current_cell = cv2.resize(current_cell, (reference_cell.shape[1], reference_cell.shape[0]))

    # Grayscale + flou
    ref_gray = cv2.cvtColor(reference_cell, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_cell, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)

    # Zone centrale
    h, w = ref_gray.shape
    ref_crop = ref_gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    curr_crop = curr_gray[h // 4:3 * h // 4, w // 4:3 * w // 4]

    # Histogramme
    hist_ref = cv2.calcHist([ref_crop], [0], None, [32], [0, 256])
    hist_ref = cv2.normalize(hist_ref, hist_ref).flatten()

    hist_curr = cv2.calcHist([curr_crop], [0], None, [32], [0, 256])
    hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()

    similarity = cv2.compareHist(hist_ref, hist_curr, cv2.HISTCMP_INTERSECT)

    # Seuil différencié
    threshold = 0.90 if is_white else 0.80  # seuil plus tolérant sur cases noires

    if debug:
        print(f"Similarity: {similarity:.3f} | {'white' if is_white else 'black'} | threshold: {threshold}")

    return similarity >= threshold



