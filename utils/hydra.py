from omegaconf import DictConfig

def replace_reference(conf):
    for k in conf:
        if isinstance(conf[k], str):
            conf[k] = conf[k]
        elif isinstance(conf[k], DictConfig):
            replace_reference(conf[k])


def fix_relative_path(conf):
    for k in conf:
        if isinstance(conf[k], str) and k.endswith("dir"):
            if not conf[k].startswith(conf.root_dir):
                conf[k] = conf.root_dir + "/" + conf[k]
        elif isinstance(conf[k], DictConfig):
            fix_relative_path(conf[k])