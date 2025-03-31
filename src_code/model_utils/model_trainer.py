import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from sentence_transformers import SentenceTransformer
from trl import SFTTrainer
import csv
import evaluate
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

class UTNChatBot():
    def __init__(self, config):
        self.config = config
        self._create_model()
        self._create_tokenizer()

    def _create_model(self):
        if os.path.exists(self.config.checkpoint):
            print("Loading model from checkpoint")
            model_path = self.config.checkpoint
        else:
            model_path = self.config.model_name
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.config.device_map,
            cache_dir=self.config.cache_dir
        )
        self.model.config.use_cache = False

    def _create_tokenizer(self):
        if os.path.exists(self.config.checkpoint):
            tokenizer_path = self.config.checkpoint
        else:
            tokenizer_path = self.config.tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            device_map=self.config.device_map,
            cache_dir=self.config.cache_dir
        )
    
    def _set_fine_tuning_parameters(self):
        self.training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim=self.config.optim,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.lr,
            weight_decay=self.config.weight_decay,
            max_steps=self.config.max_steps,
            warmup_ratio=self.config.warmup_ratio,
            group_by_length=self.config.group_by_length,
            lr_scheduler_type=self.config.lr_scheduler_type,
            do_eval=self.config.do_eval,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            logging_strategy=self.config.logging_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            logging_dir=os.path.join(self.config.output_dir, "../logs"),
            report_to="tensorboard"
        )

    def train(self, dataset, plot=True):
        self.model.train()
        self._set_fine_tuning_parameters()
        # Trainer API
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            processing_class=self.tokenizer,
        )

        # Start training
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.config.output_dir, "checkpoint_final"))

        if plot:
            log_history = self.trainer.state.log_history
            self._plot_loss(log_history) 

    def inference(self, prompt, system_context="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def infer_batch(self, dataset, result_path=None):
        # Data to be written to the CSV file
        data = [
            ['Prompt', 'ground_truth', 'response', 'bleu_score', 'rouge_score', 'cosine_similarities']
        ]
        sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=self.config.cache_dir)
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load('meteor')
        for i in tqdm(range(len(dataset['test']))):
            prompt = dataset['test'][i]['messages'][1]['content']
            system_context = dataset['test'][i]['messages'][0]['content']
            response = self.inference(prompt, system_context=system_context)
            ground_truth = dataset['test'][i]['messages'][2]['content']
            # Calculate BLEU score
            bleu_score = bleu_metric.compute(predictions=[response], references=[ground_truth])
            # Calculate ROUGE score
            rouge_score = rouge_metric.compute(predictions=[response], references=[ground_truth])
            # Calculate METEOR score
            meteor_score = meteor_metric.compute(predictions=[response], references=[ground_truth])
            # Calculate Semantic Similarity
            embeddings1 = sentence_transformer_model.encode([response], convert_to_tensor=True)
            embeddings2 = sentence_transformer_model.encode([ground_truth], convert_to_tensor=True)
            cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
            cosine_similarities = torch.mean(cosine_similarities)
            data.append([prompt, ground_truth, response, bleu_score, rouge_score, cosine_similarities.item()])
        # Open the CSV file in write mode
        if result_path is None:
            result_path = f"{self.config.output_dir}/../metrics_result/results.csv"
        with open(f"{result_path}", mode="w", newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            # Write the data to the CSV file
            writer.writerows(data)
        print(f"CSV file has been written successfully to {self.config.output_dir}/output.csv.")

    def evaluate(self, dataset, result_path=None):
        sentence_transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=self.config.cache_dir)
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load('meteor')
        refs = []
        responses = []
        for i in range(len(dataset['test'])):
            prompt = dataset['test'][i]['messages'][1]['content']
            system_context = dataset['test'][i]['messages'][0]['content']
            ground_truth = dataset['test'][i]['messages'][2]['content']
            response = self.inference(prompt, system_context=system_context)
            refs.append(ground_truth)
            responses.append(response)
        
        # Calculate BLEU score
        bleu_score = bleu_metric.compute(predictions=responses, references=refs)
        print(f"BLEU Score: {bleu_score}")
        # Calculate ROUGE score
        rouge_score = rouge_metric.compute(predictions=responses, references=refs)
        print(f"ROUGE Score: {rouge_score}")
        # Calculate METEOR score
        meteor_score = meteor_metric.compute(predictions=responses, references=refs)
        print(f"METEOR Score: {meteor_score}")
        # Calculate Semantic Similarity
        embeddings1 = sentence_transformer_model.encode(responses, convert_to_tensor=True)
        embeddings2 = sentence_transformer_model.encode(refs, convert_to_tensor=True)
        cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        cosine_similarities = torch.mean(cosine_similarities)
        print(f"Semantic Similarity: {cosine_similarities.item()}")

        results = {"BLEU": bleu_score, "ROUGE": rouge_score, "METEOR": meteor_score, "Semantic Similarity": cosine_similarities.item()}
        if result_path is None:
            result_path = f"{self.config.output_dir}/../metrics_result/results.json"
        with open(f"{result_path}", "w") as f:
            json.dump(results, f, indent=4)
        return results

    def _plot_loss(self, log_history):
        train_steps = []
        train_loss = []
        eval_steps = []
        eval_loss = []

        for log in log_history:
            if "loss" in log and "step" in log:
                train_steps.append(log["step"])
                train_loss.append(log["loss"])
            if "eval_loss" in log and "step" in log:
                eval_steps.append(log["step"])
                eval_loss.append(log["eval_loss"])

        plt.figure(figsize=(10, 5))
        if train_loss:
            plt.plot(train_steps, train_loss, label="Training Loss")
        if eval_loss:
            plt.plot(eval_steps, eval_loss, label="Evaluation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training & Evaluation Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "../images", "loss_plot.png"))
        plt.show()