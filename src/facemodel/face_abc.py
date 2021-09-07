import abc


class Face(abc.ABC):

    @abc.abstractmethod
    def show(self, img_path, args):
        print(img_path, args)
        return NotImplemented
