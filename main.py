from src_code.task_utils.config_parser import ConfigParser
from src_code.data_utils.dataset import dataset_generator
from src_code.model_utils.model_trainer import UTNChatBot
import sys

def main(config_path) -> None:
    # all the parameters can be obtained from this configs object
    configs: ConfigParser = ConfigParser(config_path).get_parser()
    print(f"{configs.batch_size = }")

    print("### Creating Dataloaders ###")
    final_dataset = dataset_generator(configs.data_path)
    UTN_chat_bot = UTNChatBot(configs)

    

    if configs.task == 'train':
        print("### Training Model ###")
        UTN_chat_bot.train(final_dataset)

    elif configs.task == 'evaluate':
        # use the checkpoint to evaluate the model and get a score
        raise Exception(f'This is yet to be implemented!')

    elif configs.task == 'inference':
        responses = UTN_chat_bot.inference("Hello")
        print(f"{responses=}")

    elif configs.task == 'inference_full':
        UTN_chat_bot.infer_batch(final_dataset)
        
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
