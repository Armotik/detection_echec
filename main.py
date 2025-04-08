import cv2
import numpy as np

from board_detection import compute_chessboard_perspective, draw_chessboard, extract_cells_from_lines
from pieces_detection import is_cell_empty

class Projet:

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.perspective_matrix = None
        self.size = 400

    def detect_movement(self, frame):
        # https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html
        return frame

    def run(self):

        # cap = cv2.VideoCapture('data/echiquier_vide_1.wm')
        cap = cv2.VideoCapture('data/mat_du_berger_1.wm')

        while True:
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if self.perspective_matrix is None:
                self.perspective_matrix = compute_chessboard_perspective(frame)

            if self.perspective_matrix is not None:
                rectified = cv2.warpPerspective(frame, self.perspective_matrix, (self.size, self.size))

                # Appliquer le damier
                rectified, verticals, horizontals = draw_chessboard(rectified)

                cells = extract_cells_from_lines(rectified, verticals, horizontals)

                for (i, j), cell_img in cells:
                    # Coordonnées de la cellule
                    x1 = verticals[j][0]
                    x2 = verticals[j + 1][0]
                    y1 = horizontals[i][1]
                    y2 = horizontals[i + 1][1]

                    # Centrer le texte
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    label = "vide" if is_cell_empty(((i, j), cell_img), threshold=0.8) else "pleine"

                    color = (0, 255, 0) if label == "pleine" else (0, 0, 255)

                    cv2.putText(rectified, label, (center_x - 20, center_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                cv2.imshow('Frame', rectified)

            else:
                cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    projet = Projet()
    projet.run()
