from collections import deque

import cv2
import numpy as np


def bg_substractor(frame: np.ndarray, fgbg: cv2.BackgroundSubtractor, kernel: np.ndarray) -> np.ndarray:
    """
    Applique un filtre de soustraction d'arrière-plan à l'image donnée.
    :param frame: Image d'entrée
    :param fgbg: Objet de soustraction d'arrière-plan
    :param kernel: Noyau pour l'opération morphologique
    :return: Masque binaire après soustraction d'arrière-plan
    """
    # https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    return fgmask

def is_someone_moving(mask: np.ndarray, min_area: int = 500) -> bool:
    """
    Vérifie si un mouvement est détecté dans le masque binaire.
    :param mask: Masque binaire généré par le bg_substractor
    :param min_area: Aire minimale pour considérer un mouvement valide
    :return: True si un mouvement est détecté, False sinon
    """
    # Détection des contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            return True  # Un mouvement valide a été détecté

    return False  # Aucun mouvement valide détecté

def detect_movement_source(mask, left_zone_ratio=0.3, right_zone_ratio=0.3, min_area=5000):
    """
    Détecte si un mouvement provient de la gauche, de la droite ou des deux côtés, en filtrant les petits mouvements.
    :param mask: Masque binaire généré par le bg_substractor
    :param left_zone_ratio: Ratio de la largeur pour la zone du joueur de gauche
    :param right_zone_ratio: Ratio de la largeur pour la zone du joueur de droite
    :param min_area: Aire minimale pour considérer un contour comme un mouvement valide
    :return: Tuple (gauche: bool, droite: bool)
    """
    height, width = mask.shape

    # Définir les zones pour chaque joueur
    left_zone_end = int(width * left_zone_ratio)
    right_zone_start = int(width * (1 - right_zone_ratio))

    # Détection des contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_detected = False
    right_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue  # Ignorer les petits mouvements

        # Obtenir le centre du contour
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2

        # Vérifier si le contour est dans la zone gauche ou droite
        if center_x < left_zone_end:
            left_detected = True
        elif center_x > right_zone_start:
            right_detected = True

    return left_detected, right_detected
