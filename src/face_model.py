import pathlib

from facemodel.lbpcascade_animeface import LibCascadeAnimeFace


class FaceModel:

    model = None

    def __init__(self, root_path, name):
        if name == "lbpcascade_animeface":
            self.model = LibCascadeAnimeFace(root_path)

    def show(self, img_path, args):
        self.model.show(img_path, args)


if __name__ == "__main__":
    root_path = pathlib.Path(__file__).parent.resolve()
    model = FaceModel(root_path, "lbpcascade_animeface")
    model.show("I:\\work\\WORK\\ReclassifyAnimeCG\\ReclassifyAnimeCG\\data-sample\\classified-data\\Emilia\\1.jpg", None)
