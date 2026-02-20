"""
Data loader for dialogue training in JSONL format.
OPTIMIZED VERSION - Faster loading, better memory management.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from functools import lru_cache
import os


class DialogueDataset(Dataset):
    """
    Dataset for dialogue training from JSONL format.

    OPTIMIZED:
    - Pre-allocated tensor storage
    - Efficient padding in collate
    - Memory-mapped loading option for huge datasets

    Format:
    {
        "context": ["User: Hello", "Assistant: Hi!", "User: How are you?"],
        "response": "I'm doing great!",
        "source": "lmsys_chat"
    }
    """
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_seq_len=256,
        cache_tokenization=True,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_tokenization = cache_tokenization
        self.pad_token_id = tokenizer.pad_token_id

        # Load examples
        print(f"📂 Loading dialogues from {jsonl_path}...")
        self.examples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="   Loading"):
                try:
                    data = json.loads(line.strip())
                    if 'context' in data and 'response' in data:
                        self.examples.append(data)
                except:
                    continue

        print(f"   ✅ Loaded {len(self.examples):,} dialogues")

        # Pre-tokenize for faster training
        if cache_tokenization:
            print(f"🔄 Pre-tokenizing examples...")
            self._preprocess_all()

    def _preprocess_all(self):
        """Pre-tokenize all examples into contiguous tensors"""
        self.tokenized_cache = []

        for example in tqdm(self.examples, desc="   Tokenizing"):
            tokens = self._tokenize_example(example)
            if tokens is not None:
                # Store as tensor directly (faster than list)
                self.tokenized_cache.append(
                    torch.tensor(tokens, dtype=torch.long)
                )

        # Free original examples if we have cache
        if self.tokenized_cache:
            self.examples = None  # Free memory

        print(f"   ✅ Cached {len(self.tokenized_cache):,} tokenized examples")

    def _tokenize_example(self, example):
        """Convert context + response to token IDs"""
        context_text = "\n".join(example['context'])
        response_text = example['response']
        full_text = f"{context_text}\n{response_text}"

        tokens = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
        )

        if len(tokens) < 2:
            return None

        return tokens

    def __len__(self):
        if self.cache_tokenization and self.tokenized_cache:
            return len(self.tokenized_cache)
        return len(self.examples)

    def __getitem__(self, idx):
        """Returns input_ids and labels tensors"""
        if self.cache_tokenization and self.tokenized_cache:
            tokens = self.tokenized_cache[idx]
        else:
            example = self.examples[idx]
            tokens = self._tokenize_example(example)

            if tokens is None:
                return self.__getitem__((idx + 1) % len(self))

            tokens = torch.tensor(tokens, dtype=torch.long)

        # Input: all tokens except last, Target: all tokens except first
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:],
        }


class CollateFn:
    """
    Picklable collate function class for Windows multiprocessing compatibility.

    Optimizations:
    - Pre-allocate output tensors
    - Use torch.full for faster initialization
    - Minimize Python loops
    """
    def __init__(self, pad_token_id=50256):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Get lengths and max length
        lengths = [item['input_ids'].size(0) for item in batch]
        max_len = max(lengths)
        batch_size = len(batch)

        # Pre-allocate tensors (faster than building lists and stacking)
        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_len),
            -100,  # Ignore index for CrossEntropyLoss
            dtype=torch.long
        )

        # Fill in actual values
        for i, (item, length) in enumerate(zip(batch, lengths)):
            input_ids[i, :length] = item['input_ids']
            labels[i, :length] = item['labels']

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


# Keep function version for backward compatibility
def optimized_collate_fn(batch, pad_token_id=50256):
    """Functional version for single-process use"""
    return CollateFn(pad_token_id)(batch)


def create_dataloader(
    jsonl_path,
    tokenizer,
    batch_size=1,
    max_seq_len=256,
    shuffle=True,
    num_workers=2,
    cache_tokenization=True,
):
    """
    Create OPTIMIZED DataLoader for dialogue training.

    Optimizations:
    - persistent_workers: Keep worker processes alive
    - prefetch_factor: Load more batches ahead
    - pin_memory: Faster GPU transfer
    - Optimized collate function

    Args:
        jsonl_path: Path to dialogues_combined.jsonl
        tokenizer: GPT-2 tokenizer
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        shuffle: Shuffle data
        num_workers: DataLoader workers (2-4 recommended)
        cache_tokenization: Pre-tokenize all examples

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = DialogueDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        cache_tokenization=cache_tokenization,
    )

    # Create picklable collate function (works on Windows with multiprocessing)
    collate_fn = CollateFn(pad_token_id=tokenizer.pad_token_id)

    # Optimized DataLoader settings
    use_workers = num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=use_workers,  # Keep workers alive between epochs
        prefetch_factor=4 if use_workers else None,  # Prefetch 4 batches per worker
        drop_last=True,  # Avoid variable batch sizes (better for compilation)
    )

    return dataloader


def get_tokenizer(tokenizer_name='gpt2'):
    """
    Load tokenizer with optimized settings.

    Supports:
    - 'gpt2' (default, 50K vocab)
    - Any HuggingFace tokenizer name (e.g., 'Qwen/Qwen-7B')
    - Local custom tokenizer directory (e.g., './custom_tokenizer')
    """
    from transformers import AutoTokenizer
    import os

    print(f"📝 Loading tokenizer: {tokenizer_name}")

    # Check if it's a local directory
    if os.path.isdir(tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            local_files_only=True,
        )
        print(f"   ✅ Loaded custom tokenizer from {tokenizer_name}")
    else:
        # Try loading from HuggingFace
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                trust_remote_code=True,  # Needed for some tokenizers like Qwen
            )
        except Exception as e:
            print(f"   ⚠️  Failed to load {tokenizer_name}, falling back to gpt2")
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Add a pad token if none exists
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    print(f"   ✅ Vocab size: {len(tokenizer):,}")

    return tokenizer


# === Benchmark utilities ===

def benchmark_dataloader(dataloader, num_batches=100):
    """Benchmark dataloader throughput"""
    import time

    # Warmup
    iterator = iter(dataloader)
    for _ in range(min(5, len(dataloader))):
        next(iterator)

    # Timed run
    iterator = iter(dataloader)
    start = time.perf_counter()

    for i, batch in enumerate(iterator):
        if i >= num_batches:
            break
        # Simulate GPU transfer
        _ = batch['input_ids'].cuda(non_blocking=True)
        _ = batch['labels'].cuda(non_blocking=True)

    elapsed = time.perf_counter() - start
    batches_per_sec = min(num_batches, len(dataloader)) / elapsed

    return {
        'batches_per_second': batches_per_sec,
        'ms_per_batch': 1000 / batches_per_sec,
        'elapsed': elapsed,
    }


if __name__ == "__main__":
    print("🔬 Testing OPTIMIZED DialogueDataset...\n")

    # Load tokenizer
    tokenizer = get_tokenizer()
    print(f"✅ Loaded tokenizer (vocab size: {len(tokenizer)})")

    # Create sample JSONL
    import tempfile

    sample_data = [
        {
            "context": ["User: Hello", "Assistant: Hi there!"],
            "response": "How can I help you?",
            "source": "test"
        },
        {
            "context": ["User: What's the weather?"],
            "response": "I don't have access to weather data, but you can check online!",
            "source": "test"
        },
    ] * 100  # More samples for realistic test

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name

    # Create dataset
    dataset = DialogueDataset(
        jsonl_path=temp_path,
        tokenizer=tokenizer,
        max_seq_len=128,
        cache_tokenization=True,
    )

    # Test first example
    print(f"\n📋 First example:")
    sample = dataset[0]
    print(f"   Input shape: {sample['input_ids'].shape}")
    print(f"   Label shape: {sample['labels'].shape}")

    # Create dataloader (num_workers=0 for test compatibility on Windows)
    dataloader = create_dataloader(
        jsonl_path=temp_path,
        tokenizer=tokenizer,
        batch_size=4,
        max_seq_len=128,
        num_workers=0,  # Use 0 for testing; increase for actual training
    )

    print(f"\n📦 DataLoader test:")
    batch = next(iter(dataloader))
    print(f"   Batch input shape: {batch['input_ids'].shape}")
    print(f"   Batch labels shape: {batch['labels'].shape}")

    # Benchmark (if CUDA available)
    if torch.cuda.is_available():
        print(f"\n⏱️  Benchmarking DataLoader...")
        stats = benchmark_dataloader(dataloader, num_batches=50)
        print(f"   Throughput: {stats['batches_per_second']:.1f} batches/sec")
        print(f"   Latency: {stats['ms_per_batch']:.2f} ms/batch")

    print(f"\n✅ Data loading works!")

    # Cleanup
    os.unlink(temp_path)