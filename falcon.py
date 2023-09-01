from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

#model = "tiiuae/falcon-40b-instruct"
model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)




while True:
    entered_text = input("PROMPT:")
    sequences = pipeline(
        entered_text,
        max_length=400,
        do_sample=True,
        top_k=20,
        num_return_sequences=3,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")




