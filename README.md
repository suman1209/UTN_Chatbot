# UTN_Chatbot
An LLM for answering UTN academics related question

## Env Setting

```
pip install -r requirements.txt
```

## run main file
choose the task from "./configs/configs_simple.yaml" and run the following

```python
python3 main.py configs/configs_simple.yaml
```

## To test a checkpoint performance/outputs
check the notebook at notebooks/submission_test.ipynb
## Tensorboard

```
tensorboard --logdir docs_andpython3 main.py configs/configs_simple.yaml_results/logs/
```
