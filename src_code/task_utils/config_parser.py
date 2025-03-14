from pathlib import Path
import sys
import yaml


class ConfigParser:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    @classmethod
    def get_object(cls):
        if cls._instance is None:
            raise RuntimeError("You need to instantiate this class first")
        return cls._instance

    def __init__(self, config_path):
        self.config_dict = self.get_config(config_path)

    def get_parser(self):
        # ADD NEW CONFIG PARAMETERS HERE
        self.task = self.config_dict.get("task")
        # data configs
        data_configs = self.config_dict.get("data_configs")
        if data_configs is None:
            raise Exception("data_configs is not available!")
        self.train_path = data_configs.get("train_path")
        self.val_path = data_configs.get("train_path")
        self.test_path = data_configs.get("train_path")
        self.shuffle = data_configs.get("shuffle")
        self.batch_size = data_configs.get("batch_size")

        # model configs
        model_configs = self.config_dict.get("model_configs")
        if model_configs is None:
            raise KeyError("model_configs is missing from the config file!")
        self.model_name = model_configs.get("name")
        self.save_checkpoint = model_configs.get("save_checkpoint", None)
        self.resume_from_checkpoint = model_configs.get("resume_from_checkpoint", None)
        self.device = model_configs.get('device')
        self.print_freq = model_configs.get('print_freq')
        self.epochs = model_configs.get("epochs")

        optim_configs = model_configs.get("optim")
        self.lr = optim_configs.get("lr")
        self.momentum = optim_configs.get("momentum")
        self.weight_decay = optim_configs.get("weight_decay")
        self.clip_grad = optim_configs.get("clip_grad")

        # task_configs
        task_configs = self.config_dict.get("task_configs")
        self.debug = task_configs.get("debug")
        self.log_expt = task_configs.get("log_expt")
        self.vocab_size = task_configs.get("vocab_size")
        return self

    def update(self, additional_config: dict):
        for key, val in additional_config.items():
            setattr(self, key, val)

    def __verify__argparse(self, config_path):
        if isinstance(config_path, str):
            return config_path
        elif config_path is None:
            args_count = len(sys.argv)
            if (args_count) > 2:
                print(f"One argument expected, got {args_count - 1}")
                raise SystemExit(2)
            elif args_count <= 1:
                print("You must specify the config file")
                raise SystemExit(2)

            config_path = Path(sys.argv[1])
            return config_path
        elif isinstance(config_path, dict):
            return config_path
        else:
            raise Exception(f"{config_path = }")
        print(f"{config_path } is being used!")

    def get_config(self, config):
        config = self.__verify__argparse(config)
        print(f"{config = }")
        if isinstance(config, (str, Path)):
            # reading from yaml config file
            with open(config, 'r') as file:
                config_dict = yaml.safe_load(file)
        elif isinstance(config, dict):
            config_dict = config
        return config_dict
