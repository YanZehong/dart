_modules = {}

def register_module(name):

    def decorator(cls):
        _modules[name] = cls
        return cls

    return decorator


def build_module(**kwargs):
    assert "conf" in kwargs
    conf = kwargs["conf"]

    return _modules[conf.task](**kwargs)