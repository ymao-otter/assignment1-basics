#!/usr/bin/env python3
"""
Text generation script using trained Transformer model.
"""
import argparse
import sys
from pathlib import Path

import torch

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from bpe.bpe_tokenizer import Tokenizer
from transformer.model import TransformerLM
from transformer.serialization import load_checkpoint


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device: str = 'cuda'
) -> str:
    """Generate text from a prompt using the model.
    
    Args:
        model: Trained transformer model
        tokenizer: BPE tokenizer
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter (0 = no top-k)
        device: Device to run on
        
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get predictions for last token
            # Crop to context length if needed
            context = input_ids[:, -model.context_length:]
            
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for end token
            if next_token.item() == tokenizer.token_to_id.get(b'<|endoftext|>', -1):
                break
    
    # Decode
    generated_tokens = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='data/tinystories_vocab.json',
                        help='Path to vocab file')
    parser.add_argument('--merges', type=str, default='data/tinystories_merges.txt',
                        help='Path to merges file')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (0.1-2.0, higher=more random)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling (0=disabled)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TEXT GENERATION WITH TRAINED TRANSFORMER")
    print("=" * 70)
    
    # Load tokenizer
    print(f"\n1. Loading tokenizer from {args.vocab}...")
    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=['<|endoftext|>']
    )
    print(f"   ✅ Vocab size: {len(tokenizer.vocab)}")
    
    # Initialize model
    print(f"\n2. Initializing model...")
    # These should match your training config
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000.0,
        context_length=512,
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model parameters: {num_params:,}")
    
    # Load checkpoint
    print(f"\n3. Loading checkpoint from {args.checkpoint}...")
    iter_num = load_checkpoint(args.checkpoint, model, optimizer=None)
    print(f"   ✅ Loaded checkpoint from iteration {iter_num}")
    
    # Generate samples
    print(f"\n4. Generating {args.num_samples} sample(s)...")
    print(f"   Prompt: '{args.prompt}'")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Top-k: {args.top_k}")
    
    print("\n" + "=" * 70)
    
    for i in range(args.num_samples):
        print(f"\nSample {i+1}:")
        print("-" * 70)
        
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
        
        print(generated)
        print()
    
    print("=" * 70)
    print("Generation complete!")


if __name__ == '__main__':
    main()
