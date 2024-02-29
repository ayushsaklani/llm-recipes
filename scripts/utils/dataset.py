from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from utils.constants import *
from utils.util import *

class Dataset:
    def __init__(self,dataset_config,tokenizer_config,seed:99) -> None:
        print(dataset_config)
        self.text_filed = dataset_config.dataset_text_field
        self.padding_side = tokenizer_config.padding_side
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.model_name)
        self.data = load_dataset(dataset_config.name,trust_remote_code=True)

    def generate_prompt(self,
        context: str, question: str,answer:str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
         ) -> str:
        empty_str =" "
        msg = [{"role":"system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role":"user","content": f'{clean_text(question) if question else empty_str}'},
        {"role":"context","content":f'{clean_text(context) if context else empty_str}'},
        {"role":"assistant","content":clean_text(answer) if answer  else empty_str}
        ]
        return self.tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=False)
    
    def generate_text(self,data_point):
        return {"text":self.generate_prompt(context=data_point["context"],question=data_point["instruction"],answer=data_point["response"])}

    def process_dataset(self,data: Dataset):
        return (
            data.shuffle(seed=self.seed)
            .map(self.generate_text)
            .remove_columns("category")
        )

    def prepare_dataset(self):
        self.data["train"] = self.process_dataset(self.data["train"])
        self.data["validation"] = self.process_dataset(self.data["validation"])
        return self.data