from transformers import AutoTokenizer, TextStreamer
from models.modeling_llama import LlamaForCausalLM

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = [{"role": "system", "content": "Act as a scientist"},
        {"role":"user", "content": "what is a black hole?"}]

inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt = True)
streamer = TextStreamer(tokenizer, skip_prompt=True)

model.generate(inputs, streamer = streamer, max_length=256)