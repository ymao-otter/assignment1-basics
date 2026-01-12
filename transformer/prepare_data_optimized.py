"""
Memory-optimized data preparation script for training.

This script tokenizes text data using a BPE tokenizer and saves it as a binary
file for memory-efficient training with np.memmap.

Key optimizations:
- Processes chunks in streaming fashion with limited queue
- Writes results incrementally to temporary files
- Minimal memory footprint regardless of dataset size
"""

import argparse
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bpe.bpe_tokenizer import Tokenizer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _encode_chunk(
    chunk_data: tuple[str, str, str, list[str] | None, int]
) -> tuple[int, bytes]:
    """
    Worker function to encode a text chunk.
    
    Args:
        chunk_data: Tuple of (text_chunk, vocab_path, merges_path, special_tokens, chunk_id)
    
    Returns:
        Tuple of (chunk_id, token_ids_as_bytes)
    """
    text_chunk, vocab_path, merges_path, special_tokens, chunk_id = chunk_data
    
    # Each worker creates its own tokenizer instance
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens
    )
    
    # Encode the chunk
    token_ids = tokenizer.encode(text_chunk)
    
    # Convert to numpy array immediately and return as bytes
    token_array = np.array(token_ids, dtype=np.uint16)
    return chunk_id, token_array.tobytes()


def chunk_generator(input_path: str, chunk_size: int):
    """Generator that yields text chunks from input file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        chunk_id = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk_id, chunk
            chunk_id += 1


def prepare_dataset(
    input_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] | None = None,
    chunk_size: int = 1000000,
    num_workers: int | None = None,
    max_queue_size: int = 50,
) -> None:
    """
    Tokenize a text file and save as binary format (memory-optimized).
    
    Args:
        input_path: Path to input text file
        output_path: Path to output binary file
        vocab_path: Path to vocabulary JSON file
        merges_path: Path to merges text file
        special_tokens: Optional list of special tokens
        chunk_size: Number of characters to process at once per chunk
        num_workers: Number of parallel workers (default: CPU count)
        max_queue_size: Max chunks to submit at once (controls memory usage)
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    logger.info(f"Reading {input_path}...")
    
    # Get file size for progress reporting
    file_size = os.path.getsize(input_path)
    logger.info(f"File size: {file_size:,} bytes")
    logger.info(f"Using {num_workers} workers with max queue size {max_queue_size}")
    
    # Create a temporary directory for chunk files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"Using temporary directory: {temp_dir}")
        
        total_chunks = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create chunk generator
            chunks = chunk_generator(input_path, chunk_size)
            
            # Submit initial batch
            futures = {}
            for _ in range(max_queue_size):
                try:
                    chunk_id, text_chunk = next(chunks)
                    chunk_data = (text_chunk, vocab_path, merges_path, special_tokens, chunk_id)
                    future = executor.submit(_encode_chunk, chunk_data)
                    futures[future] = chunk_id
                except StopIteration:
                    break
            
            if not futures:
                logger.error("No chunks to process!")
                return
            
            total_expected = len(futures)
            
            with tqdm(desc='Tokenizing', unit='chunk') as pbar:
                # Process results as they complete and submit new work
                while futures:
                    # Wait for at least one future to complete
                    done_set, pending_set = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    
                    # Process all completed futures
                    for done in done_set:
                        chunk_idx = futures.pop(done)
                        try:
                            returned_id, token_bytes = done.result()
                            
                            # Write to temporary file immediately
                            chunk_file = temp_path / f"chunk_{returned_id:06d}.bin"
                            with open(chunk_file, 'wb') as f:
                                f.write(token_bytes)
                            
                            total_chunks += 1
                            pbar.update(1)
                            
                            # Submit next chunk if available
                            try:
                                chunk_id, text_chunk = next(chunks)
                                chunk_data = (text_chunk, vocab_path, merges_path, special_tokens, chunk_id)
                                new_future = executor.submit(_encode_chunk, chunk_data)
                                futures[new_future] = chunk_id
                                pbar.total = total_chunks + len(futures)
                                pbar.refresh()
                                
                            except StopIteration:
                                pass  # No more chunks to submit
                                
                        except Exception as e:
                            logger.error(f"Error processing chunk {chunk_idx}: {e}")
                            raise
        
        logger.info(f"Processed {total_chunks} chunks")
        logger.info("Combining chunks into final file...")
        
        # Combine all chunk files in order
        chunk_files = sorted(temp_path.glob("chunk_*.bin"))
        total_tokens = 0
        
        with open(output_path, 'wb') as output_file:
            for chunk_file in tqdm(chunk_files, desc='Combining', unit='chunk'):
                with open(chunk_file, 'rb') as f:
                    chunk_data = f.read()
                    output_file.write(chunk_data)
                    total_tokens += len(chunk_data) // 2  # uint16 = 2 bytes per token
        
        logger.info(f"Total tokens: {total_tokens:,}")
    
    # Verify the file can be memory-mapped
    logger.info("Verifying saved data...")
    data = np.memmap(output_path, dtype=np.uint16, mode='r')
    logger.info(f"Verification successful. Data shape: {data.shape}")
    logger.info(f"First 10 tokens: {data[:10].tolist()}")
    
    logger.info("Done!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare text data for training (memory-optimized)"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input text file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output binary file'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocabulary JSON file'
    )
    parser.add_argument(
        '--merges',
        type=str,
        required=True,
        help='Path to merges text file'
    )
    parser.add_argument(
        '--special-tokens',
        type=str,
        nargs='+',
        default=None,
        help='List of special tokens'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000000,
        help='Number of characters to process at once'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--max-queue-size',
        type=int,
        default=50,
        help='Maximum chunks in processing queue (controls memory usage)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        vocab_path=args.vocab,
        merges_path=args.merges,
        special_tokens=args.special_tokens,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        max_queue_size=args.max_queue_size,
    )


if __name__ == '__main__':
    main()
