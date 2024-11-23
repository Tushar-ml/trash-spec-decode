import json
from tqdm import tqdm

def convert_from_sharegpt_format(data_json: str, output_file_json: str) -> None:
    
    inputs = json.load(open(data_json, 'r'))
    formatted_json = []
    
    for data in tqdm(inputs):
        conversation = data["conversations"]
        local_conv = []
        for conv in conversation:
            conv_formatted = {}
            if conv["from"] == "human":
                conv_formatted["role"] = "user"
            elif conv["from"] == "gpt":
                conv_formatted["role"] = "assistant"
            
            conv_formatted["content"] = conv["value"]
            local_conv.append(conv_formatted)
        formatted_json.append(local_conv)
    
    json.dump(formatted_json, open(output_file_json, 'w'))


if __name__ == "__main__":
    
    convert_from_sharegpt_format("data/ShareGPT_V3_unfiltered_cleaned_split.json", "data/sharegpt_common.json")