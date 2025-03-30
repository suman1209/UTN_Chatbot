from src_code.task_utils.config_parser import ConfigParser
from src_code.data_utils.dataset import dataset_generator
from src_code.model_utils.model_trainer import UTNChatBot
import sys

def main(config_path) -> None:
    # all the parameters can be obtained from this configs object
    configs: ConfigParser = ConfigParser(config_path).get_parser()
    print(f"{configs.batch_size = }")

    print("### Creating Dataloaders ###")

    final_dataset = dataset_generator(configs.data_path, configs.sys_role, shuffle=configs.shuffle)
    test_dataset = dataset_generator("./datasets/LLM Project - Test_Questions.tsv", configs.sys_role, train_ratio=0, val_ratio=0, shuffle=False)

    UTN_chat_bot = UTNChatBot(configs)

    

    if configs.task == 'train':
        print("### Training Model ###")
        UTN_chat_bot.train(final_dataset, configs.plot)

    elif configs.task == 'evaluate':
        UTN_chat_bot.evaluate(test_dataset)

    elif configs.task == 'inference':
        responses = UTN_chat_bot.inference("Hello")
        print(f"{responses=}")

    elif configs.task == 'inference_full':
        UTN_chat_bot.infer_batch(test_dataset)

    elif configs.task == 'interactive':
        print("### Interactive Mode ###")
        print("Press Ctrl+C to exit")
        while True:
            prompt = input("User: ")
            response = UTN_chat_bot.inference(prompt, system_context=configs.sys_role)
            print(f"Assistant: {response}")
        
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
