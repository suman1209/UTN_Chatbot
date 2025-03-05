from ..task_utils.config_parser import ConfigParser

def get_dataloader():
    configs = ConfigParser.get_object()
    # you can use the parameters from the configs file like below
    print(f"Batch_size from dataloader script: {configs.batch_size}")
    pass