# import json
# from tqdm import tqdm

# def convert_from_sharegpt_format(data_json: str, output_file_json: str) -> None:
    
#     inputs = json.load(open(data_json, 'r'))
#     formatted_json = []
    
#     for data in tqdm(inputs):
#         conversation = data["conversations"]
#         local_conv = []
#         for conv in conversation:
#             conv_formatted = {}
#             if conv["from"] == "human":
#                 conv_formatted["role"] = "user"
#             elif conv["from"] == "gpt":
#                 conv_formatted["role"] = "assistant"
            
#             conv_formatted["content"] = conv["value"]
#             local_conv.append(conv_formatted)
#         formatted_json.append(local_conv)
    
#     json.dump(formatted_json, open(output_file_json, 'w'))


# if __name__ == "__main__":
    
#     convert_from_sharegpt_format("data/ShareGPT_V3_unfiltered_cleaned_split.json", "data/sharegpt_common.json")

import os
from models.token_recycling import TokenRecycling

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # must match target / tokenizer
k = 8  # must match vLLM token_recycling_k

tr = TokenRecycling(model_id, k=k)
tr.generate_matrix_using_dataset(
    dataset_path="data/your_sharegpt_as_chat.json",
    max_length=512,      # greedy continuation steps per conversation (see below)
    nsamples=100,        # or -1 for all
    save_matrix_path="matrices/sharegpt_matrix_k8.pt",
)
os.makedirs("matrices", exist_ok=True)
# save_matrix_path is written inside generate_matrix_using_dataset when set