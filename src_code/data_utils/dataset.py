from datasets import Dataset, DatasetDict
from copy import deepcopy


def dataset_generator(csv_path, shuffle=True, test_ratio=0.1):
    sys_role="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    train_list = []
    test_list = []

    message_temp = {
        "messages": [
            {
                "role": "system",
                "content": sys_role
            },
            {
                "role": "user",
                "content": None
            },
            {
                "role": "assistant",
                "content": None
            }
        ]
    }

    with open(csv_path, "r") as f:
        lines = f.readlines()
    
    shuffle(lines)
    test_size = int(len(lines) * test_ratio)
    lines = lines[:-test_size]
    for line in lines[:-test_size]:
        line = line.strip()
        if line == "":
            continue
        line = line.split(",")
        user_prompt = line[0]
        results = line[1]
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = results
        train_list.append(message)
    
    for line in lines[-test_size:]:
        line = line.strip()
        if line == "":
            continue
        line = line.split(",")
        user_prompt = line[0]
        results = line[1]
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = results
        test_list.append(message)

    
    print(f"{len(train_list)=}")
    print(f"{len(test_list)=}")
    # print(f"{train_list[0]=}")

    train_dataset = Dataset.from_list(train_list)
    test_dataset = Dataset.from_list(test_list)
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return dataset
