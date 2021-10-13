import abc


class Model(abc.ABC):

    @abc.abstractmethod
    def load_model(self, cfg):
        print(cfg)
        return NotImplemented
