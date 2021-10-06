import abc


class Model(abc.ABC):

    @abc.abstractmethod
    def init(self, args):
        print(args)
        return NotImplemented
