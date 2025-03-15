import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

class UTNChatBot():
    def __init__(self, config):
        self.config = config
        self._create_model()
        self._create_tokenizer()

    def _create_model(self):
        if os.path.exists(self.config.checkpoint):
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
        if os.path.exists(self.config.tokenizer_checkpoint):
            tokenizer_path = self.config.tokenizer_checkpoint
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
        )

    def train(self, train_dataset, eval_dataset):
        self.model.train()
        self._set_fine_tuning_parameters()
        # Trainer API
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Start training
        self.trainer.train()
        self.trainer.save_model(self.config.output_dir)

    def inference(self, prompt):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
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
    


