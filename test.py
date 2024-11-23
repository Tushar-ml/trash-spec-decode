import torch

def custom_generate(
    model,
    tokenizer,
    input_ids,
    max_length=50,
    eos_token_id=None,
    pad_token_id=None,
    device='cuda'
):
    """
    Custom implementation of sequence generation using greedy search.

    Args:
        model: The transformer model (e.g., GPT, BERT for decoding).
        tokenizer: Tokenizer for encoding/decoding.
        input_ids: Tensor of input token IDs.
        max_length: Maximum length of the output sequence.
        eos_token_id: Token ID for the end-of-sequence token.
        pad_token_id: Token ID for padding (used for incomplete sequences).
        device: Device for computation (e.g., 'cuda' or 'cpu').

    Returns:
        Generated token IDs.
    """
    model.eval()
    generated_ids = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Pass the input through the model
            outputs = model(generated_ids)
            
            # Extract logits for the last token
            logits = outputs.logits[:, -1, :]
            
            # Choose the token with the highest probability (greedy search)
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            # Append the token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # Stop if EOS token is generated
            if eos_token_id is not None and torch.all(next_token_id == eos_token_id):
                break

    # Pad sequences to max_length if necessary
    if pad_token_id is not None:
        generated_ids = torch.nn.functional.pad(
            generated_ids,
            (0, max_length - generated_ids.size(1)),
            value=pad_token_id
        )
    
    return generated_ids


# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load a model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate sequences
    generated_ids = custom_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_length=30,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode and print the result
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
