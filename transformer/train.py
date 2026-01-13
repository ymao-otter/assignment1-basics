"""
Training script for Transformer Language Model.

This script provides a complete training loop with:
- Configurable hyperparameters
- Memory-efficient data loading with np.memmap
- Checkpoint saving/loading
- Training and validation logging
- Learning rate scheduling
- Gradient clipping
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from bpe.bpe_tokenizer import Tokenizer
from transformer.data import data_loading
from transformer.model import TransformerLM
from transformer.optimizer import AdamW, get_learning_rate
from transformer.serialization import save_checkpoint, load_checkpoint


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training run."""
    
    def __init__(
        self,
        # Model hyperparameters
        vocab_size: int = 10000,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        rope_theta: float = 10000.0,
        context_length: int = 512,
        
        # Training hyperparameters
        batch_size: int = 32,
        max_iters: int = 100000,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        warmup_iters: int = 2000,
        
        # Logging and checkpointing
        log_interval: int = 100,
        eval_interval: int = 1000,
        eval_iters: int = 100,
        checkpoint_interval: int = 5000,
        checkpoint_dir: str = "checkpoints",
        
        # Data paths
        train_data_path: str = "data/train.bin",
        val_data_path: str = "data/val.bin",
        vocab_path: str = "data/vocab.json",
        merges_path: str = "data/merges.txt",
        
        # Device and precision
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision: bool = True,
        
        # Resume training
        resume_from: Optional[str] = None,
        
        # Special tokens
        special_tokens: Optional[list[str]] = None,
    ):
        # Model params
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.context_length = context_length
        
        # Training params
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        
        # Logging and checkpointing
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Data paths
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        
        # Device and precision
        self.device = device
        self.mixed_precision = mixed_precision and device == "cuda"
        
        # Resume
        self.resume_from = resume_from
        
        # Special tokens
        self.special_tokens = special_tokens
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save config to JSON file."""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def __repr__(self) -> str:
        """String representation of config."""
        config_str = "TrainingConfig:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"  {key}: {value}\n"
        return config_str


class Trainer:
    """Trainer class for managing training loop."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Save config
        config.to_json(os.path.join(config.checkpoint_dir, "config.json"))
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_files(
            config.vocab_path,
            config.merges_path,
            special_tokens=config.special_tokens
        )
        
        # Update vocab size from tokenizer
        self.config.vocab_size = len(self.tokenizer.vocab)
        
        # Load data with memory mapping
        logger.info("Loading training data with memory mapping...")
        self.train_data = self._load_data_memmap(config.train_data_path)
        logger.info(f"Training data loaded: {len(self.train_data):,} tokens")
        
        logger.info("Loading validation data with memory mapping...")
        self.val_data = self._load_data_memmap(config.val_data_path)
        logger.info(f"Validation data loaded: {len(self.val_data):,} tokens")
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = TransformerLM(
            vocab_size=self.config.vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            context_length=config.context_length,
        ).to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model initialized with {num_params:,} parameters")
        
        # Initialize optimizer
        logger.info("Initializing optimizer...")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler('cuda') if config.mixed_precision else None
        
        # Initialize training state
        self.iter_num = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
    
    def _load_data_memmap(self, data_path: str) -> np.memmap:
        """Load data using memory mapping for efficiency."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                "Please prepare your data as a binary file of token IDs."
            )
        
        # Load as memory-mapped array
        # Assuming data is stored as uint16 (supports vocab size up to 65536)
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        return data
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and resume training."""
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        self.iter_num = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer
        )
        logger.info(f"Resumed from iteration {self.iter_num}")
    
    def _save_checkpoint(self, checkpoint_path: str) -> None:
        """Save checkpoint."""
        logger.info(f"Saving checkpoint to {checkpoint_path}...")
        save_checkpoint(
            self.model,
            self.optimizer,
            self.iter_num,
            checkpoint_path
        )
    
    def _get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        data = self.train_data if split == 'train' else self.val_data
        inputs, targets = data_loading(
            data,
            self.config.batch_size,
            self.config.context_length,
            str(self.device)
        )
        return inputs.long(), targets.long()
    
    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Estimate loss on train and validation sets."""
        self.model.eval()
        out = {}
        
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                inputs, targets = self._get_batch(split)
                
                if self.config.mixed_precision:
                    with autocast('cuda'):
                        logits = self.model(inputs)
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        )
                else:
                    logits = self.model(inputs)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                
                losses[k] = loss.item()
            
            out[split] = losses.mean().item()
        
        self.model.train()
        return out
    
    def _compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        if self.config.mixed_precision:
            with autocast('cuda'):
                logits = self.model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
        else:
            logits = self.model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return loss
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Configuration:\n{self.config}")
        
        self.model.train()
        
        # Training metrics
        running_loss = 0.0
        t0 = time.time()
        
        while self.iter_num < self.config.max_iters:
            # Update learning rate
            lr = get_learning_rate(
                self.iter_num,
                self.config.learning_rate,
                self.config.min_learning_rate,
                self.config.warmup_iters,
                self.config.max_iters
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Get batch
            inputs, targets = self._get_batch('train')
            
            # Forward pass
            loss = self._compute_loss(inputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            self.iter_num += 1
            
            # Logging
            if self.iter_num % self.config.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                avg_loss = running_loss / self.config.log_interval
                tokens_per_sec = (
                    self.config.batch_size * 
                    self.config.context_length * 
                    self.config.log_interval / dt
                )
                
                logger.info(
                    f"iter {self.iter_num:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"tokens/sec {tokens_per_sec:.0f}"
                )
                
                running_loss = 0.0
                t0 = time.time()
            
            # Evaluation
            if self.iter_num % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                logger.info(
                    f"iter {self.iter_num:6d} | "
                    f"train loss {losses['train']:.4f} | "
                    f"val loss {losses['val']:.4f}"
                )
                
                # Save best model
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    best_path = os.path.join(
                        self.config.checkpoint_dir,
                        "best_model.pt"
                    )
                    self._save_checkpoint(best_path)
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            
            # Checkpointing
            if self.iter_num % self.config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_iter_{self.iter_num}.pt"
                )
                self._save_checkpoint(checkpoint_path)
        
        # Save final checkpoint
        final_path = os.path.join(
            self.config.checkpoint_dir,
            "final_model.pt"
        )
        self._save_checkpoint(final_path)
        logger.info("Training complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model"
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file'
    )
    
    # Model hyperparameters
    parser.add_argument('--vocab-size', type=int, default=10000)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--rope-theta', type=float, default=10000.0)
    parser.add_argument('--context-length', type=int, default=512)
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-iters', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--min-learning-rate', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-iters', type=int, default=2000)
    
    # Logging and checkpointing
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--eval-iters', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=5000)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    
    # Data paths
    parser.add_argument('--train-data-path', type=str, default='data/train.bin')
    parser.add_argument('--val-data-path', type=str, default='data/val.bin')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.json')
    parser.add_argument('--merges-path', type=str, default='data/merges.txt')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision training')
    
    # Resume
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Special tokens
    parser.add_argument('--special-tokens', type=str, nargs='+', default=None, help='List of special tokens')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config from file if provided, otherwise use command line args
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            grad_clip=args.grad_clip,
            warmup_iters=args.warmup_iters,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            vocab_path=args.vocab_path,
            merges_path=args.merges_path,
            device=args.device,
            mixed_precision=not args.no_mixed_precision,
            resume_from=args.resume_from,
            special_tokens=args.special_tokens,
        )
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
