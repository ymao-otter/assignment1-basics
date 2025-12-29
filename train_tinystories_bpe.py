#!/usr/bin/env python3
"""Train BPE tokenizer on TinyStories dataset and save to files.

Memory-efficient version using chunked file reading (memory mapping approach)
to process the full dataset without loading it entirely into RAM.
"""
import time
from pathlib import Path
from bpe.bpe_training import train_bpe_to_files


def main():
    # Paths
    data_dir = Path(__file__).parent / "data"
    input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    vocab_output = data_dir / "tinystories_vocab.json"
    merges_output = data_dir / "tinystories_merges.txt"

    # Training parameters
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # Check if input file exists
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        print("\nPlease download the dataset first:")
        print("  cd data")
        print("  curl -L -o TinyStoriesV2-GPT4-train.txt \\")
        print(
            "    https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
        )
        return

    print("=" * 70)
    print("Training BPE Tokenizer on TinyStories Dataset (Full)")
    print("=" * 70)
    print("\n💾 Memory-efficient mode: Using chunked file reading")
    print("   (Memory-mapped approach - doesn't load entire file into RAM)")
    print(f"\nInput file: {input_path}")

    # Get file size
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Dataset size: {file_size_mb:.2f}MB")

    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"\nOutput files:")
    print(f"  - Vocabulary: {vocab_output}")
    print(f"  - Merges: {merges_output}")

    print("\n⏳ Starting training on full dataset...")
    start_time = time.time()

    train_bpe_to_files(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        vocab_output_path=vocab_output,
        merges_output_path=merges_output,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n✅ Training completed successfully!")
    print(f"⏱️  Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"\n📄 Output files saved:")
    print(f"  - {vocab_output}")
    print(f"  - {merges_output}")
    print(f"\n💾 Processed {file_size_mb:.2f}MB using memory-efficient chunking")
    print("=" * 70)


if __name__ == "__main__":
    main()
