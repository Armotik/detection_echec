import cv2
import numpy as np
import math

from numpy import ndarray

SIZE = 400

def order_points(pts : np.ndarray) -> np.ndarray:
    """
    Trie les points d'un rectangle dans l'ordre : top-left, top-right, bottom-right, bottom-left.
    :param pts: 4 points du rectangle
    :return: 4 points triés
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def deduplicate(lines : list[tuple[any, any, any, any]], axis=0, threshold=10):
    """
    Supprime les lignes dupliquées en fonction de leur position sur l'axe spécifié.
    :param lines: Liste de lignes à dédupliquer
    :param axis: 0 pour l'axe x, 1 pour l'axe y
    :param threshold: Distance minimale pour considérer deux lignes comme différentes
    :return: Liste de lignes dédupliquées
    """
    lines_sorted = sorted(lines, key=lambda l: l[axis])
    dedup = []
    for l in lines_sorted:
        if not dedup or abs(l[axis] - dedup[-1][axis]) > threshold:
            dedup.append(l)
    return dedup


def compute_chessboard_perspective(image: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de transformation perspective pour un échiquier.
    :param image: Image d'entrée
    :return: Matrice de transformation perspective
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    edges = cv2.Canny(thresh, 30, 100)

    # Dilatation pour combler les trous (pièces, mains)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_debug = image.copy()
    h, w = image.shape[:2]
    image_area = h * w

    for contour in contours:
        area = cv2.contourArea(contour)

        # On ne garde que les contours de taille raisonnable
        if area < 0.01 * image_area or area > 0.9 * image_area:
            continue

        # On ne garde que les contours convexes
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si on a bien 4 coins, parfait
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = np.array([p[0] for p in approx], dtype='float32')
            rect = order_points(pts)
        else:
            # Sinon on tente une reconstruction rectangulaire via minAreaRect
            rect_box = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect_box)
            rect = order_points(np.array(box, dtype='float32'))

        # Dessin debug
        cv2.drawContours(image_debug, [np.int32(rect)], -1, (255, 0, 0), 2)
        for p in rect:
            cv2.circle(image_debug, tuple(p.astype(int)), 5, (0, 255, 0), -1)

        dst_pts = np.array([
            [0, 0],
            [SIZE - 1, 0],
            [SIZE - 1, SIZE - 1],
            [0, SIZE - 1]
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(rect, dst_pts)

        cv2.imshow("DEBUG - contour choisi", image_debug)
        cv2.waitKey(1)

        return M

    # Aucun contour valide trouvé
    cv2.imshow("DEBUG - tous les candidats", image_debug)
    cv2.waitKey(1)

    return None


def draw_chessboard(rectified : np.ndarray) -> tuple[np.ndarray, list[tuple[any, any, any, any]], list[tuple[any, any, any, any]]]:
    """
    Dessine le damier sur l'image rectifiée et détecte les lignes horizontales et verticales.
    :param rectified: Image rectifiée
    :return: Image avec damier, lignes verticales, lignes horizontales
    """
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=rectified.shape[1] // 4,
        maxLineGap=10
    )

    verticals = []
    horizontals = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Vertical ≈ +/-90°
            if abs(angle) > 80:
                verticals.append((x1, y1, x2, y2))
            # Horizontal ≈ 0°
            elif abs(angle) < 10:
                horizontals.append((x1, y1, x2, y2))

    verticals = deduplicate(verticals, axis=0)
    horizontals = deduplicate(horizontals, axis=1)

    for x1, y1, x2, y2 in verticals + horizontals:
        cv2.line(rectified, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return rectified, verticals, horizontals


def extract_cells_from_lines(rectified : ndarray, verticals : list[tuple[any, any, any, any]], horizontals : list[tuple[any, any, any, any]]) -> list[tuple[tuple[int, int], ndarray]]:
    """
    Extrait les cellules du damier à partir des lignes détectées.
    :param rectified: Image rectifiée
    :param verticals: Lignes verticales détectées
    :param horizontals: Lignes horizontales détectées
    :return: Liste de cellules extraites sous forme de tuples (coordonnées, image de la cellule)
    """
    verticals = sorted(verticals, key=lambda l: l[0])
    horizontals = sorted(horizontals, key=lambda l: l[1])

    cells = []
    for i in range(len(horizontals) - 1):
        for j in range(len(verticals) - 1):
            x1 = verticals[j][0]
            x2 = verticals[j + 1][0]
            y1 = horizontals[i][1]
            y2 = horizontals[i + 1][1]

            cell = rectified[y1:y2, x1:x2]
            cells.append(((i, j), cell))

    return cells
