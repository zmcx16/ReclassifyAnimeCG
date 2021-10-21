import pathlib
import yaml


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cfg_path=""):
        if cfg_path == "":
            root = pathlib.Path(__file__).parent.resolve()
            self.cfg_path = root / "config.yaml"
        else:
            self.cfg_path = cfg_path

        with open(self.cfg_path, "r", encoding="utf-8") as stream:
            self.cfg = yaml.load(stream, Loader=yaml.FullLoader)

    def show(self):
        print(self.cfg)

    def get(self):
        return self.cfg


if __name__ == "__main__":
    print('load config')
    cfg = Config()
    cfg.show()
