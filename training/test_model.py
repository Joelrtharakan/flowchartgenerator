import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
LORA_PATH = "./phi3-finetuned"

# -----------------------
# LOAD TOKENIZER
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -----------------------
# LOAD MODEL (4-bit QLoRA)
# -----------------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# -----------------------
# GENERATE FUNCTION
# -----------------------
def generate(prompt):
    # Structured prompt (important)
    text = f"<|user|>\nCreate a clean Mermaid flowchart with Start and End: {prompt}\n<|end|>\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,  # deterministic output
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------
# TEST
# -----------------------
if __name__ == "__main__":
    test_prompt = "deploy model in Azure"

    result = generate(test_prompt)

    print("\n=== MODEL OUTPUT ===\n")
    print(result)
