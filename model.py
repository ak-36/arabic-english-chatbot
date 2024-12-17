from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import os
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAVE_PATH = "./local_models/meta-llama/Llama-3.1-8B-Instruct"  # A folder in your repo
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
if __name__ == '__main__':
    # Download and save the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"Model and tokenizer have been saved to {SAVE_PATH}.")
