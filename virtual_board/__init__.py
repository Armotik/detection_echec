import numpy as np
import cv2

from pieces_detection import is_cell_white, is_cell_empty


def init_board(cells: list[tuple[tuple[int, int], np.ndarray]]):
    """
    Initialise le plateau d'échecs.
    :param cells: Liste de cellules extraites sous forme de tuples (coordonnées, image de la cellule)
    :return: Matrice représentant le plateau d'échecs
    """
    board = np.empty((8, 8), dtype=object)  # Crée une matrice vide de type object

    for (i, j), cell_img in cells:
        is_white = is_cell_white(((i, j), cell_img))
        is_empty = is_cell_empty(cell_img, is_white, 0.5)

        cell_info = {
            "coord": (i, j),
            "img": cell_img,
            "is_white": is_white,
            "is_empty": is_empty,
        }
        board[i, j] = cell_info  # Remplit la cellule avec le dictionnaire

    return board


def update_board(board: np.ndarray, cells: list[tuple[tuple[int, int], np.ndarray]]) -> np.ndarray:
    """
    Met à jour le plateau d'échecs avec de nouvelles cellules.
    :param board: Matrice représentant le plateau d'échecs
    :param cells: Liste de cellules extraites sous forme de tuples (coordonnées, image de la cellule)
    :return: Matrice mise à jour représentant le plateau d'échecs
    """
    for (i, j), cell_img in cells:

        cell = board[i, j]
        if isinstance(cell, dict) and "is_white" in cell and cell["is_white"] is not None:

            cell_info = {
                "coord": (i, j),
                "img": cell_img,
                "is_white": board[i, j]["is_white"],  # Conserve la couleur de la cellule
                "is_empty": is_cell_empty(cell_img, is_cell_white(((i, j), cell_img)), 0.5)
            }
            board[i, j] = cell_info  # Met à jour la cellule avec le dictionnaire

        else:

            is_white = is_cell_white(((i, j), cell_img))
            is_empty = is_cell_empty(cell_img, is_white, 0.5)

            cell_info = {
                "coord": (i, j),
                "img": cell_img,
                "is_white": is_white,
                "is_empty": is_empty,
            }
            board[i, j] = cell_info

    return board
