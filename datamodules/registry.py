_datamodules = {}


def register_datamodule(name):

    def decorator(cls):
        _datamodules[name] = cls
        return cls

    return decorator


def build_datamodule(**kwargs):
    assert "conf" in kwargs
    conf = kwargs["conf"]

    return _datamodules[conf.data.name](**kwargs)