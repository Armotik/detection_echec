import cv2
import numpy as np


class Projet:

    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.perspective_matrix = None

    def detect_movement(self, frame):
        # https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html
        return frame

    def detect_chessboard(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_cnt = None

        for cnt in contours:

            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if area > max_area and len(approx) == 4:
                max_area = area
                best_cnt = approx

        pts = best_cnt.reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        size = 400
        dst_pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype='float32')

        # Appliquer transformation
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped_board = cv2.warpPerspective(frame, M, (size, size))

        return cropped_board

    def run(self):

        while True:
            cap = cv2.VideoCapture('data/echiquier_vide_1.wm')


            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.detect_chessboard(frame)

                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

if __name__ == "__main__":
    projet = Projet()
    projet.run()
