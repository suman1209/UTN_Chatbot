# UTN_Chatbot
An LLM for answering UTN academics related question

## Environment Setting

We developed the code on Python 3.10.12
```
pip install -r requirements.txt
```

## Download model
This will save the model check points to  `./docs_and_results/checkpoints/`

```
python3 Dwonload.py
```

## Run main file

With this command it will have the interactive mode, that you can chat with the UTNChatbot
```python
python3 main.py
```


You can also change the task in the configs file, but it should be the same format as [config_file](./configs/configs_simple.yaml) and run the following

```python
python3 main.py configs/"your_configs_file_name".yaml
```

## To test a checkpoint performance/outputs
check the notebook at notebooks/submission_test.ipynb

## Tensorboard

```
tensorboard --logdir docs_andpython3 main.py configs/configs_simple.yaml_results/logs/
```
