_collators = {}

def register_collator(name):

    def decorator(cls):
        _collators[name] = cls
        return cls

    return decorator


def build_collator(**kwargs):
    assert "conf" in kwargs
    conf = kwargs["conf"]

    return _collators[conf.model.collator](**kwargs)