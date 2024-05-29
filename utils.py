import importlib

def instantiate_class(module_name, class_name, **kwargs):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls(**kwargs)
    return instance

def create_environment(env_name):
    from env import Environment
    return Environment(env_name)