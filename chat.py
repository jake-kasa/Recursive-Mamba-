"""
Chat interface for RecursiveStateSpace model.
Usage: python chat.py --checkpoint checkpoints/final.pt --tokenizer custom_tokenizer
"""

import torch
import torch.nn.functional as F
import argparse
from model import create_model
from data_loader import get_tokenizer


def load_model(checkpoint_path, tokenizer, model_args, device='cuda'):
    """Load model from checkpoint"""
    print(f"📂 Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or use provided args
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"   ✅ Found config in checkpoint")
    else:
        # Use config from command line arguments
        config = {
            'n_blocks': model_args.n_blocks,
            'recurrent_depth': model_args.recurrent_depth,
            'd_model': model_args.d_model,
            'd_state': model_args.d_state,
            'd_conv': model_args.d_conv,
            'expand_factor': model_args.expand_factor,
            'vocab_size': len(tokenizer),
            'max_seq_len': model_args.max_seq_len,
            'use_checkpointing': False,
        }
        print(f"   ⚠️  No config in checkpoint, using command line args")

    # Override vocab size to match tokenizer
    config['vocab_size'] = len(tokenizer)
    config['use_checkpointing'] = False

    print(f"   Config: d_model={config['d_model']}, n_blocks={config['n_blocks']}, depth={config['recurrent_depth']}")

    model = create_model(config, compile_model=False)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"   ✅ Model loaded!")
    return model


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    device='cuda'
):
    """Generate text from a prompt"""

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    # Get special token IDs
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.encode('\n')[-1] if tokenizer.encode('\n') else None

    for _ in range(max_new_tokens):
        # Truncate if too long for model
        if generated.shape[1] > model.max_seq_len:
            context = generated[:, -model.max_seq_len:]
        else:
            context = generated

        # Forward pass
        logits = model(context)
        next_token_logits = logits[0, -1, :]

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
            next_token_logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Stop at EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # Stop at newline after "Assistant:" response (basic stopping)
        decoded_so_far = tokenizer.decode(generated[0], skip_special_tokens=True)
        if decoded_so_far.count('\n') > prompt.count('\n') + 3:
            break

    # Decode and return only the new part
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return full_text


def chat_loop(model, tokenizer, device='cuda'):
    """Interactive chat loop"""

    print("\n" + "=" * 60)
    print("💬 Chat Interface")
    print("=" * 60)
    print("Type your message and press Enter.")
    print("Commands: /quit, /clear, /temp <value>, /tokens <value>")
    print("=" * 60 + "\n")

    conversation_history = []
    temperature = 0.8
    max_tokens = 200

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith('/'):
            cmd = user_input.lower().split()

            if cmd[0] == '/quit':
                print("👋 Goodbye!")
                break

            elif cmd[0] == '/clear':
                conversation_history = []
                print("🗑️  Conversation cleared.\n")
                continue

            elif cmd[0] == '/temp' and len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    print(f"🌡️  Temperature set to {temperature}\n")
                except ValueError:
                    print("❌ Invalid temperature value\n")
                continue

            elif cmd[0] == '/tokens' and len(cmd) > 1:
                try:
                    max_tokens = int(cmd[1])
                    print(f"📏 Max tokens set to {max_tokens}\n")
                except ValueError:
                    print("❌ Invalid token value\n")
                continue

            else:
                print("❓ Unknown command. Use /quit, /clear, /temp <value>, /tokens <value>\n")
                continue

        # Build prompt
        conversation_history.append(f"User: {user_input}")
        prompt = "\n".join(conversation_history) + "\nAssistant:"

        # Generate response
        print("Assistant: ", end="", flush=True)

        try:
            response = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device=device
            )

            # Extract just the assistant's response
            if "Assistant:" in response:
                assistant_reply = response.split("Assistant:")[-1].strip()
            else:
                assistant_reply = response[len(prompt):].strip()

            # Clean up the response (stop at next "User:" if present)
            if "User:" in assistant_reply:
                assistant_reply = assistant_reply.split("User:")[0].strip()

            print(assistant_reply)

            # Add to history
            conversation_history.append(f"Assistant: {assistant_reply}")

        except Exception as e:
            print(f"\n❌ Error generating response: {e}")

        print()  # Blank line between turns


def main():
    parser = argparse.ArgumentParser(description='Chat with your trained model')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='custom_tokenizer',
                        help='Tokenizer name or path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Max tokens to generate')

    # Model architecture args (must match training!)
    parser.add_argument('--n_blocks', type=int, default=48,
                        help='Number of Mamba blocks (must match training)')
    parser.add_argument('--recurrent_depth', type=int, default=5,
                        help='Recurrent depth (must match training)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension (must match training)')
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Convolution kernel size')
    parser.add_argument('--expand_factor', type=int, default=2,
                        help='Expansion factor')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Max sequence length')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Load tokenizer
    print(f"📝 Loading tokenizer: {args.tokenizer}")
    tokenizer = get_tokenizer(args.tokenizer)

    # Load model
    model = load_model(args.checkpoint, tokenizer, args, args.device)

    # Start chat
    chat_loop(model, tokenizer, args.device)


if __name__ == "__main__":
    main()