"""
Training script for HybridRecursiveStateSpace (Mamba + Attention).
OPTIMIZED VERSION - torch.compile, fused ops, reduced sync points.
Optimized for RTX 2060 (6GB VRAM) with 48 Mamba blocks + 6 attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CUDA OPTIMIZATIONS - Must be set BEFORE any CUDA operations
# ============================================================================
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmuls (Ampere+)
torch.backends.cudnn.allow_tf32 = True  # TF32 for cuDNN
torch.set_float32_matmul_precision('high')  # Use TF32 precision

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import os
import time
import json
from tqdm import tqdm
import argparse

from model_hybrid import create_hybrid_model
from data_loader import create_dataloader, get_tokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def generate_sample(model, tokenizer, prompt, max_tokens=50, temperature=0.8, device='cuda'):
    """Quick generation for testing during training"""
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(max_tokens):
            logits = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature

            # Top-k sampling
            top_k = 50
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    model.train()
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Strip the prompt to show only the response
    return generated_text[len(prompt):]


class OptimizedTrainer:
    """
    OPTIMIZED Trainer for HybridRecursiveStateSpace.

    Optimizations:
    - Reduced CUDA sync points (no empty_cache every 10 steps)
    - Fused optimizer step with gradient unscaling
    - Pre-allocated metrics storage
    - Efficient logging
    """
    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        scaler,
        tokenizer,
        device='cuda',
        accumulation_steps=16,
        max_grad_norm=1.0,
        checkpoint_dir='checkpoints_hybrid',
        log_interval=100,
        save_interval=5000,
        test_interval=2000,
        test_prompts=None,
        empty_cache_interval=500,  # Much less frequent cache clearing
        scheduler=None,  # Learning rate scheduler
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scaler = scaler
        self.tokenizer = tokenizer
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.test_interval = test_interval
        self.empty_cache_interval = empty_cache_interval
        self.scheduler = scheduler

        # Default test prompts
        if test_prompts is None:
            self.test_prompts = [
                "User: Hello! How are you?\nAssistant:",
                "User: What is 2+2?\nAssistant:",
                "User: Tell me a fun fact.\nAssistant:",
            ]
        else:
            self.test_prompts = test_prompts

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Pre-allocate metrics lists with estimated capacity
        estimated_steps = 100000
        self.metrics = {
            'train_loss': [],
            'steps': [],
            'learning_rates': [],
            'memory_used': [],
        }

    def test_generation(self):
        """Run test generation to monitor model progress"""
        print(f"\n{'=' * 60}")
        print(f"🧪 Test Generation (Step {self.global_step})")
        print(f"{'=' * 60}")

        for prompt in self.test_prompts:
            response = generate_sample(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=50,
                temperature=0.8,
                device=self.device
            )
            # Only show the generated part, not the prompt
            generated_part = response[len(prompt):] if response.startswith(prompt) else response
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_part}")

        print(f"{'=' * 60}\n")

    def train_epoch(self, epoch):
        """Train for one epoch with all optimizations"""
        self.model.train()
        self.epoch = epoch

        epoch_loss = 0.0
        num_batches = 0

        # Pre-compute inverse for efficiency
        inv_accumulation = 1.0 / self.accumulation_steps

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Non-blocking GPU transfer (data should already be pinned)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast('cuda'):
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                # Scale loss for accumulation
                scaled_loss = loss * inv_accumulation

            # Backward pass with gradient scaling
            self.scaler.scale(scaled_loss).backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale, clip, step, update - all in sequence
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster

                # Step the learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                # Sparse cache clearing (every 500 steps instead of 10)
                if self.global_step % self.empty_cache_interval == 0:
                    torch.cuda.empty_cache()

                # Logging
                if self.global_step % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    mem_allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                    self.metrics['train_loss'].append(loss.item())
                    self.metrics['steps'].append(self.global_step)
                    self.metrics['learning_rates'].append(current_lr)
                    self.metrics['memory_used'].append(mem_allocated)

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.2e}',
                        'mem': f'{mem_allocated:.2f}GB',
                        'step': self.global_step,
                    })

                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')

                # Test generation
                if self.test_interval > 0 and self.global_step % self.test_interval == 0:
                    self.test_generation()

            epoch_loss += loss.item()
            num_batches += 1

        return epoch_loss / num_batches

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{name}.pt')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'metrics': self.metrics,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")

        # Also save metrics separately
        metrics_path = os.path.join(self.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"\n📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.metrics = checkpoint['metrics']

        print(f"   ✅ Resumed from step {self.global_step}, epoch {self.epoch}")


def create_optimizer(model, args):
    """Create optimizer with optional 8-bit or fused variants"""

    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases, layer norms, embeddings
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    # Try 8-bit AdamW first (most memory efficient)
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            print(f"   ✅ Using 8-bit AdamW (bitsandbytes)")
            return optimizer
        except ImportError:
            print(f"   ⚠️  bitsandbytes not available, falling back to fused AdamW")

    # Try fused AdamW (faster than standard)
    if args.use_fused_adam:
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=True,
            )
            print(f"   ✅ Using fused AdamW")
            return optimizer
        except:
            print(f"   ⚠️  Fused AdamW not available, using standard AdamW")

    # Standard AdamW fallback
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    print(f"   ✅ Using standard AdamW")
    return optimizer


def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")

    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Tokenizer
    print(f"\n🔤 Loading tokenizer...")
    tokenizer = get_tokenizer(args.tokenizer)
    print(f"   ✅ Tokenizer loaded: {args.tokenizer}")

    # Model configuration
    model_config = {
        'n_blocks': args.n_blocks,
        'attention_interval': args.attention_interval,
        'n_attention_heads': args.n_attention_heads,
        'recurrent_depth': args.recurrent_depth,
        'd_model': args.d_model,
        'd_state': args.d_state,
        'd_conv': args.d_conv,
        'expand_factor': args.expand_factor,
        'vocab_size': len(tokenizer),
        'max_seq_len': args.max_seq_len,
        'use_checkpointing': args.use_checkpointing,
        'checkpoint_segments': args.checkpoint_segments,
        'dropout': args.dropout,
    }

    # Create model
    print(f"\n🤖 Creating hybrid model...")
    model = create_hybrid_model(
        config=model_config,
        compile_model=args.use_compile,
    )
    model = model.to(device)

    if args.use_compile and hasattr(torch, 'compile'):
        print(f"   ⚠️  Model will compile on first forward pass")
        print(f"   Note: First forward pass will trigger actual compilation")

    # Data loader
    print(f"\n📂 Creating data loader...")
    train_loader = create_dataloader(
        jsonl_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        num_workers=args.num_workers,
        cache_tokenization=args.cache_tokenization,
    )

    total_steps = len(train_loader) * args.epochs
    print(f"   ✅ {len(train_loader):,} batches per epoch")
    print(f"   Total steps: {total_steps:,}")

    # Optimizer
    print(f"\n⚙️  Setting up optimizer...")
    optimizer = create_optimizer(model, args)

    # Learning rate scheduler (warmup + cosine decay)
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)  # Cap warmup at 10% of training

    if warmup_steps > 0:
        print(f"\n📈 Setting up learning rate scheduler...")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Total steps: {total_steps}")
        print(f"   Min LR: {args.min_lr}")

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=args.min_lr
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        print(f"   ✅ Warmup + Cosine decay scheduler ready")
    else:
        scheduler = None
        print(f"\n📈 No learning rate scheduler (warmup_steps=0)")

    # Gradient scaler
    try:
        scaler = GradScaler('cuda')
    except TypeError:
        scaler = GradScaler()

    # Trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        tokenizer=tokenizer,
        device=device,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        test_interval=args.test_interval,
        empty_cache_interval=args.empty_cache_interval,
        scheduler=scheduler,
    )

    # Resume from checkpoint
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)

    # Print training config
    print(f"\n🏋️  Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Accumulation steps: {args.accumulation_steps}")
    print(f"   Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"   Learning rate: {args.learning_rate} → {args.min_lr} (cosine decay)")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Empty cache interval: {args.empty_cache_interval} steps")
    if args.test_interval > 0:
        print(f"   🧪 Test generation every {args.test_interval} batches")
    print()

    # Warmup run to trigger compilation
    if args.use_compile and hasattr(torch, 'compile'):
        print("🔥 Warmup run (triggers torch.compile)...")
        warmup_batch = next(iter(train_loader))
        warmup_input = warmup_batch['input_ids'].to(device)
        with torch.no_grad(), autocast('cuda'):
            _ = model(warmup_input)
        print("   ✅ Model compiled\n")

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = trainer.train_epoch(epoch)
        trainer.save_checkpoint(f'epoch_{epoch}')

        elapsed = time.time() - start_time
        epochs_done = epoch + 1
        epochs_remaining = args.epochs - epochs_done
        eta = (elapsed / epochs_done) * epochs_remaining if epochs_done > 0 else 0

        print(f"⏱️  Elapsed: {elapsed/3600:.1f}h, ETA: {eta/3600:.1f}h\n")

    # Final save
    trainer.save_checkpoint('final')

    total_time = time.time() - start_time
    print(f"\n✅ Training complete!")
    print(f"   Total time: {total_time/3600:.2f} hours")
    print(f"   Final loss: {epoch_loss:.4f}")
    print(f"   Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hybrid RecursiveStateSpace (Mamba + Attention)')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dialogues JSONL file')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer name')

    # Model
    parser.add_argument('--n_blocks', type=int, default=48,
                        help='Number of physical Mamba blocks')
    parser.add_argument('--attention_interval', type=int, default=8,
                        help='Insert attention layer every N Mamba blocks (1:N ratio)')
    parser.add_argument('--n_attention_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--recurrent_depth', type=int, default=5,
                        help='Recursive depth')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Convolution kernel size')
    parser.add_argument('--expand_factor', type=int, default=2,
                        help='Expansion factor in blocks')
    parser.add_argument('--max_seq_len', type=int, default=448,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for attention blocks')

    # Training
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=6,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate (for cosine decay)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps (0 to disable scheduler)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm')

    # Optimization flags (use --no-* to disable, e.g., --no-use_compile)
    parser.add_argument('--use_checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Use gradient checkpointing')
    parser.add_argument('--checkpoint_segments', type=int, default=8,
                        help='Blocks per checkpoint segment')
    parser.add_argument('--use_8bit_adam', action=argparse.BooleanOptionalAction, default=False,
                        help='Use 8-bit AdamW (requires bitsandbytes)')
    parser.add_argument('--use_fused_adam', action=argparse.BooleanOptionalAction, default=True,
                        help='Use fused AdamW (PyTorch 2.0+)')
    parser.add_argument('--use_compile', action=argparse.BooleanOptionalAction, default=False,
                        help='Use torch.compile (Linux only, requires Triton)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--cache_tokenization', action=argparse.BooleanOptionalAction, default=True,
                        help='Pre-tokenize all examples')
    parser.add_argument('--empty_cache_interval', type=int, default=500,
                        help='Clear CUDA cache every N optimizer steps')

    # Logging & Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_hybrid',
                        help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Save checkpoint every N batches')
    parser.add_argument('--test_interval', type=int, default=500,
                        help='Test generation every N batches (0 to disable)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint')

    # DataLoader
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers')

    args = parser.parse_args()

    main(args)
