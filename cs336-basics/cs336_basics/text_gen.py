import argparse
import torch
import jaxtyping
from jaxtyping import Float

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Load the model
parser = argparse.ArgumentParser(description="Training LLM")

parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
parser.add_argument("--input-text", type=str, required=True, help="Input text for generation.")
parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling.")
parser.add_argument("--top-p", type=float, default=0.95, help="Top-P (nucleus) sampling.")
parser.add_argument("--device", type=str, default="cpu", help="Torch device string, e.g., 'cuda', 'cpu', 'mps'.")
parser.add_argument("--dtype", type=str, default="float32", choices=DTYPE_DICT.keys(), help="Torch dtype string, e.g., 'float32', 'bfloat16'.")
parser.add_argument("--vocab-path", type=str, required=True, help="Path to the tokenizer vocabulary file.")
parser.add_argument("--merges-path", type=str, required=True, help="Path to the tokenizer merges file.")
# Model hyperparameters needed for reconstruction
parser.add_argument("--vocab-size", type=int, required=True, help="Tokenizer vocabulary size.")
parser.add_argument("--context-length", type=int, required=True, help="Model context length.")
parser.add_argument("--num-layers", type=int, required=True, help="Number of transformer layers.")
parser.add_argument("--d-model", type=int, required=True, help="Model hidden dimension.")
parser.add_argument("--num-heads", type=int, required=True, help="Number of attention heads.")
parser.add_argument("--d-ff", type=int, required=True, help="Feed-forward hidden dimension.")
parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter.")

args = parser.parse_args()
VOCAB_PATH = args.vocab_path
MERGES_PATH = args.merges_path
MODEL_CHECKPOINT = args.model_checkpoint
INPUT_TEXT = args.input_text
MAX_NEW_TOKENS = args.max_new_tokens
TEMPERATURE = args.temperature
TOP_K = args.top_k
TOP_P = args.top_p
DEVICE = args.device
DTYPE = DTYPE_DICT[args.dtype]
VOCAB_SIZE = args.vocab_size
CONTEXT_LENGTH = args.context_length
NUM_LAYERS = args.num_layers
D_MODEL = args.d_model
NUM_HEADS = args.num_heads
D_FF = args.d_ff
ROPE_THETA = args.rope_theta
print("Using device:", DEVICE)
print("Using dtype:", DTYPE)

from cs336_basics.train.checkpointing import load_checkpoint
from cs336_basics.lm import TransformerLM
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
toeknizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])
model = TransformerLM(VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, ROPE_THETA,
                         device=DEVICE, dtype=DTYPE)
# Load the model checkpoint
if DEVICE == "cuda":
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cuda")
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(torch.load(MODEL_CHECKPOINT)["model"])
model.eval()

def generate_next(model, tokenizer, input_text, temperature, top_p, device):
    """Generate text using the model and tokenizer."""
    # tokenize the input text
    tokenized_input = torch.tensor(tokenizer.encode(input_text), device=device, dtype=torch.long).unsqueeze(0)  # Shape: [1, seq_len]
    # Generate the next token
    outlogits: Float[torch.Tensor, f"seq_len vocab_size"]
    outlogits = model.forward(tokenized_input).squeeze(0)
    # Select the prediction of the next token
    next_vocab_logits = outlogits[-1, :]
    # Apply temperature softmax
    temperature_softmax = (torch.exp(next_vocab_logits/temperature))/(torch.sum(torch.exp(next_vocab_logits/temperature)))
    # Apply top-p filtering
    sorted_probs, sorted_indices = torch.sort(temperature_softmax, descending=True) # Sort probabilities descending
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # Compute the cumulative probabilities
    top_p_indices = sorted_indices[cumulative_probs <= top_p] # Keep until cumulative prob > top_p
    if top_p_indices.numel() == 0:
        top_p_indices = sorted_indices[:1]
    # Sample from the filtered distribution
    top_p_distribution = temperature_softmax[top_p_indices]/temperature_softmax[top_p_indices].sum()
    next_token = torch.multinomial(top_p_distribution, num_samples=1)
    generated_token = [top_p_indices[next_token].item()]
    return generated_token

def generate_text(model, tokenizer, input_text, max_new_tokens, temperature, top_p, device, EOS="<|endoftext|>"):
    """Generate text using the model and tokenizer."""
    generated_tokens = []
    current_input = input_text
    for _ in range(max_new_tokens):
        next_token = generate_next(model, tokenizer, current_input, temperature, top_p, device)
        generated_tokens.extend(next_token)
        if tokenizer.decode(next_token) == EOS:
            break
        current_input += tokenizer.decode(next_token)
    return current_input



def main():
    model.eval()
    model.to(DEVICE)
    for i in range(5):
        generated_text = generate_text(model, toeknizer, INPUT_TEXT, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, DEVICE)
        print(f"Generated Text {i}:")
        print(f"============================")
        print(generated_text)

if __name__ == "__main__":
    main()
