import numpy as np
import cv2


def virtual_board(cells: list[tuple[tuple[int, int], np.ndarray]]):
    """
    Représente sous forme de matrice le plateau d'échecs pour chaque frame.
    :param cells: Liste de cellules extraites sous forme de tuples (coordonnées, image de la cellule)
    :return: Matrice représentant le plateau d'échecs
    """
    board = np.zeros((8, 8), dtype=int)

    for (i, j), cell_img in cells:
        # On peut ajouter une logique pour déterminer si la cellule est vide ou pleine
        # Par exemple, on pourrait utiliser une fonction de détection de pièces ici
        # Pour l'instant, on va juste marquer toutes les cellules comme pleines (1)
        board[i][j] = 1

    # Utilisation de connectedComponents pour identifier les composants connectés
    num_labels, labels = cv2.connectedComponents(board.astype(np.uint8))

    return board, num_labels, labels
