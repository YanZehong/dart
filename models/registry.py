_models = {}

def register_model(name):
    def decorator(cls):
        _models[name] = cls
        return cls
    
    return decorator

def build_model(**kwargs):
    assert "conf" in kwargs
    conf = kwargs["conf"]

    return _models[conf.model.arch](**kwargs)