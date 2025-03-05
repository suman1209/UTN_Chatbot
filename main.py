from src_code.task_utils.config_parser import ConfigParser
from src_code.data_utils.dataloader import get_dataloader
import sys

def main(config_path) -> None:
    # all the parameters can be obtained from this configs object
    configs: ConfigParser = ConfigParser(config_path).get_parser()
    print(f"{configs.batch_size = }")

    print("### Creating Dataloaders ###")
    get_dataloader()
    train_loader = None
    val_loader = None
    test_loader = None

    print("### Training Model ###")

    if configs.task == 'train':
        pass

    elif configs.task == 'evaluate':
        # use the checkpoint to evaluate the model and get a score
        raise Exception(f'This is yet to be implemented!')

    else:
        raise Exception(f'Undefined task! {configs.task}')

if __name__ == "__main__":
    config_path = None
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: python main.py <config_path>")
            sys.exit(1)
        config_path = sys.argv[1]  # Read config path from command line
    main(config_path)
