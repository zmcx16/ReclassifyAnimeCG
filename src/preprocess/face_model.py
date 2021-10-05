from face.lbpcascade_animeface import LibCascadeAnimeFace


class FaceModel:

    model = None

    def __init__(self, name):
        if name == "lbpcascade_animeface":
            self.model = LibCascadeAnimeFace()

    def show(self, img_path, args):
        self.model.show(img_path, args)


if __name__ == "__main__":
    model = FaceModel("lbpcascade_animeface")
    model.show("I:\\work\\WORK\\ReclassifyAnimeCG\\ReclassifyAnimeCG\\data-sample\\train\\Emilia\\1.jpg", None)
