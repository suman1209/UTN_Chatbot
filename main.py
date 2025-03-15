from src_code.task_utils.config_parser import ConfigParser
from src_code.data_utils.dataloader import get_dataloader
from src_code.model_utils.model_trainer import UTNChatBot
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
    UTN_chat_bot = UTNChatBot(configs)

    print("### Training Model ###")

    if configs.task == 'train':
        # UTN_chat_bot.train(train_loader, val_loader)
        pass

    elif configs.task == 'evaluate':
        # use the checkpoint to evaluate the model and get a score
        raise Exception(f'This is yet to be implemented!')
    elif configs.task == 'inference':
        # UTN_chat_bot.inference("Hello")
        # use the checkpoint to generate responses
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
