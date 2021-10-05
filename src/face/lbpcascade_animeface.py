import cv2
import os.path
from face.face_abc import Face


class LibCascadeAnimeFace(Face):

    def show(self, img_path, args):
        cascade_file_path = os.path.join(os.path.dirname(__file__), "lbpcascade_animeface.xml")
        if not os.path.isfile(cascade_file_path):
            raise RuntimeError("not found lbpcascade_animeface.xml")

        cascade = cv2.CascadeClassifier(cascade_file_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(24, 24))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("face detect", image)
        cv2.waitKey(0)
        return
