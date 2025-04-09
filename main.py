from collections import deque

import cv2
import numpy as np

from board_detection import compute_chessboard_perspective, draw_chessboard, extract_cells_from_lines
from movement_detection import bg_substractor, detect_movement_source, is_someone_moving
from pieces_detection import cell_compare, is_cell_white

class Projet:

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.perspective_matrix = None
        self.size = 400

        self.cropped_empty_board = cv2.imread('data/cropped_empty_board.png')
        self.empty_board_cells = None

    def extract_empty_board_cells(self):
        if self.empty_board_cells is not None:
            return self.empty_board_cells

        rectified, verticals, horizontals = draw_chessboard(self.cropped_empty_board)
        extracted = extract_cells_from_lines(rectified, verticals, horizontals)

        if len(extracted) == 64:
            self.empty_board_cells = {coord: img for coord, img in extracted}
            return self.empty_board_cells

        self.empty_board_cells = None
        return None

    def run(self):

        cap = cv2.VideoCapture('data/mat_du_berger_1.wm')
        # cap = cv2.VideoCapture('data/echiquier_vide_1.wm')

        self.extract_empty_board_cells()

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if self.perspective_matrix is None:
                self.perspective_matrix = compute_chessboard_perspective(frame)

            if self.perspective_matrix is not None:
                rectified = cv2.warpPerspective(frame, self.perspective_matrix, (self.size, self.size))

                cropped = rectified.copy()

                mask = bg_substractor(rectified, self.fgbg, self.kernel)

                left_detected, right_detected = detect_movement_source(mask)

                if is_someone_moving(mask):
                    if left_detected and right_detected:
                        cv2.putText(rectified, "Both players moving", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif left_detected:
                        cv2.putText(rectified, "Left player moving", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif right_detected:
                        cv2.putText(rectified, "Right player moving", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                else:

                    rectified, verticals, horizontals = draw_chessboard(rectified)
                    cells = extract_cells_from_lines(rectified, verticals, horizontals)

                    if len(cells) == 64:

                        for (i, j), cell_img in cells:
                            x1 = verticals[j][0]
                            x2 = verticals[j + 1][0]
                            y1 = horizontals[i][1]
                            y2 = horizontals[i + 1][1]

                            empty_cell_img = self.empty_board_cells.get((i, j))

                            print(f"Cell ({i}, {j})")
                            label = "vide" if cell_compare(cell_img, empty_cell_img, is_cell_white(((i, j), empty_cell_img)), True) else "piece"
                            color = (0, 255, 0) if label == "vide" else (0, 0, 255)

                            cv2.putText(rectified, label, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            cv2.rectangle(rectified, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            cv2.imwrite("data/test.png", rectified)

                cv2.imshow('rectified', rectified)

            else:
                cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    projet = Projet()
    projet.run()
