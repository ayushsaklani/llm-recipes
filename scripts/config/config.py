from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Experiment(BaseModel):
    name: str
    model_save_name: str
    push_to_hub:bool
    wandb_token:str


class Model(BaseModel):
    name: str
    torch_dtype: str


class Dataset(BaseModel):
    name: str
    dataset_text_field: str


class Tokenizer(BaseModel):
    model_name: str
    padding_side: str
    truncation_side: str
    pad_token_as_eos_token: bool


class Lora(BaseModel):
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]


class Train(BaseModel):
    task_type: str
    output_dir: str
    num_train_epochs: int
    seed: int
    max_seq_length: int
    lora: Lora
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    eval_accumulation_steps: int
    optim: str
    logging_steps: int
    learning_rate: float
    fp16: bool
    max_grad_norm: float
    evaluation_strategy: str
    eval_steps: float
    warmup_ratio: float
    save_strategy: str
    report_to: str
    lr_scheduler: str


class TrainingConfig(BaseModel):
    experiment: Experiment
    model: Model
    dataset: Dataset
    tokenizer: Tokenizer
    train: Train