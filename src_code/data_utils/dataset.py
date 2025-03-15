from datasets import Dataset, DatasetDict
from copy import deepcopy


def dataset_generator(csv_path, shuffle=True, train_ratio=0.8, val_ratio=0.1):
    
    sys_role="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    train_list = []
    val_list = []
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
        lines = f.readlines()[1:]

    if shuffle:
        # todo shuffle the list of lines
        pass

    # train data
    train_size = int(len(lines) * train_ratio)
    print(f"{train_size = }")
    train_lines = lines[:train_size]
    print(f"{len(train_lines) = }")
    for line in train_lines:
        line = line.strip()
        if line == "":
            continue
        line = line.split(",")
        user_prompt = line[1]
        results = line[2]
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = results
        train_list.append(message)
    
    val_size =   int(len(lines) * val_ratio)
    val_lines = lines[train_size: train_size + val_size]
    # val data
    for line in val_lines:
        line = line.strip()
        if line == "":
            continue
        line = line.split(",")
        user_prompt = line[0]
        results = line[1]
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = results
        val_list.append(message)
    
    test_size =   int(len(lines) * (1 - train_ratio -val_ratio))
    test_lines = lines[train_size + val_size: ]
    # test data
    for line in lines[train_size + val_size: ]:
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
    print(f"{len(val_list)=}")
    print(f"{len(test_list)=}")

    train_dataset = Dataset.from_list(train_list)
    val_dataset = Dataset.from_list(val_list)
    test_dataset = Dataset.from_list(test_list)
    dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })

    return dataset