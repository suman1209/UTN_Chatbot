from datasets import Dataset, DatasetDict
from copy import deepcopy
import random


def dataset_generator(csv_path, sys_role, shuffle=True, train_ratio=0.9, val_ratio=0.05):
    
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
        random.shuffle(lines)

    # train data
    train_size = int(len(lines) * train_ratio)
    print(f"{train_size = }")
    train_lines = lines[:train_size]
    print(f"{len(train_lines) = }")
    for line in train_lines:
        line = line.strip()
        line = line.split("\t")
        if len(line) < 2:
            continue
        user_prompt = line[0]
        results = line[1]
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = results
        train_list.append(message)
    
    val_size =   int(len(lines) * val_ratio)
    val_lines = lines[train_size: train_size + val_size]
    # val data
    for line in val_lines:
        line = line.strip()
        line = line.split("\t")
        if len(line) < 2:
            continue
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
        
        line = line.split("\t")
        if len(line) < 2:
            continue
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