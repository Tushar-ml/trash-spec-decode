from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoConfig
from loguru import logger
import torch
from typing import Optional
from tqdm import tqdm
import os
from tree import DraftTree

class TokenRecycling:
    def __init__(self, model_id: str, k: int, trash_file: Optional[str] = None) -> None:

        logger.info(f"Loading {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else  "cpu"
        self.model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype = torch.float16).to(self.device)

        self.config = AutoConfig.from_pretrained(model_id)
        self.vocab_size = self.config.vocab_size
        self.k = k
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if trash_file and os.path.exists(trash_file):
            self.adjacency_matrix = torch.load(trash_file)
        else:
            self.adjacency_matrix = torch.full((self.vocab_size, self.k), -1)

        logger.info(f"Adjacency matrix created of shape: ({self.vocab_size}, {k})")


    def generate_matrix_using_dataset(self, dataset_path: str, max_length: int = 512, nsamples: int = -1,
        save_matrix_path: Optional[str] = None):

        import json
        conversations = json.load(open(dataset_path, "r"))

        nsamples = nsamples if nsamples > 0 else len(conversations)
        for conv in conversations[:nsamples]:
            prompt = self.tokenizer.apply_chat_template(conv, add_generation_prompt = True, tokenize = False)

            self.prefill_matrix(prompt, max_length)

        if save_matrix_path:
            torch.save(self.adjacency_matrix, save_matrix_path)

    def prefill_matrix(self, prompt, max_length: int = 128) -> None:

        tokens = self.tokenizer(prompt, return_tensors = "pt", add_special_tokens = False).input_ids.to(self.device)

        with torch.no_grad():
            for _ in tqdm(range(max_length)):

                outputs = self.model(tokens)
                current_token = tokens[:, -1]

                logits = outputs.logits
                next_token_logits = logits[:, -1, :]

                top_k_values, top_k_indices = torch.topk(
                    next_token_logits,
                    k=self.k,
                    dim=-1
                )

                token_idx = current_token[0]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                tokens = torch.cat([tokens, next_token_id], dim=-1)

                self.adjacency_matrix[token_idx] = top_k_indices[0]
                print(self.tokenizer.batch_decode(tokens))


if __name__ == "__main__":

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    token_recycling = TokenRecycling(model_id, k = 5)

    token_recycling.generate_matrix_using_dataset("../data/sharegpt_common.json", nsamples=1, max_length=10)
    # token_recycling.prefill_matrix("hello")
