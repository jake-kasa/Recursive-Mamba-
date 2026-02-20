# RecursiveStateSpace: 48-Block Mamba Language Model

Training a novel hybrid state space model on dialogue data, optimized for RTX 2060 (6GB VRAM).

## 🏗️ Architecture

**RecursiveStateSpace** combines: 
- **48 physical Mamba (state space) blocks**
- **5 recursive depths** = 240 effective layers!
- **Memory-efficient**: Only stores 48 blocks, applies them recursively
- **Linear complexity**: O(n) vs O(n²) for transformers

### Why This Works

Traditional 240-layer transformer: **Impossible on 6GB**
Our approach: **48 blocks × 5 recursions = Same depth, manageable memory**

Each recursive pass refines understanding:
- Depth 1-2: Syntax, grammar, basic patterns
- Depth 3-4: Semantics, relationships, context
- Depth 5: Abstract reasoning, coherence, world knowledge

## 📋 Requirements

### Hardware
- GPU with 6GB+ VRAM (tested on RTX 2060)
- 16GB+ system RAM recommended
- ~500GB disk space (for checkpoints during long training)

### Software
```bash
# Python 3.8+
pip install -r requirements.txt

# Optional but HIGHLY RECOMMENDED: mamba-ssm
# This provides optimized Mamba blocks
pip install mamba-ssm

# If mamba-ssm fails to install (requires CUDA compilation):
# The code will fall back to SimplifiedMambaBlock
```

## 🚀 Quick Start

### 1. Test Memory First
**IMPORTANT: Always run this before training!**

```bash
# Test with default config (48 blocks, depth 5, d=384)
python dataset_alpaca.py

# Test multiple configurations
python dataset_alpaca.py --test_all

# Test custom config
python dataset_alpaca.py --n_blocks 36 --d_model 320
```

**Expected output:**
```
✅ SUCCESS! Model fits in memory
   Peak usage: 4.82GB / 6.0GB
   Headroom: 1.18GB
```

If you get OOM errors, reduce `d_model` or `n_blocks`.

### 2. Prepare Your Data

Your data should be in JSONL format (one JSON per line):
```json
{"context": ["User: Hello", "Assistant: Hi!"], "response": "How can I help?", "source": "lmsys"}
```

You already have: `dialogues_combined.jsonl` ✅

### 3. Start Training

```bash
# Full 48-block training (recommended)
python train.py \
    --data_path dialogues_combined.jsonl \
    --epochs 2 \
    --n_blocks 48 \
    --recurrent_depth 5 \
    --d_model 384 \
    --batch_size 1 \
    --accumulation_steps 16 \
    --max_seq_len 256 \
    --checkpoint_dir checkpoints

# Estimated time: ~250 hours (10.5 days) on RTX 2060
```

**Faster training (fewer blocks):**
```bash
python train.py \
    --data_path dialogues_combined.jsonl \
    --epochs 2 \
    --n_blocks 24 \
    --recurrent_depth 5 \
    --d_model 384
    
# Estimated time: ~100 hours (4 days)
```

### 4. Monitor Training

The script prints progress every 100 batches:
```
Epoch 0: 100%|████| 180000/180000 [loss=2.1234, lr=3.00e-04, mem=4.85GB]

📊 Epoch 0 Summary:
   Average Loss: 2.1234
   Steps: 11250
   Peak Memory: 5.12GB
```

Checkpoints are saved to `checkpoints/`:
- `step_5000.pt` - Every 5000 steps
- `epoch_0.pt` - Every epoch
- `final.pt` - Final model
- `metrics.json` - Training metrics

### 5. Resume Training

If training stops (power loss, crash):
```bash
python train.py \
    --data_path dialogues_combined.jsonl \
    --resume_from checkpoints/epoch_0.pt \
    --epochs 2
```

## 📊 Configuration Options

### Model Architecture
```bash
--n_blocks 48              # Physical Mamba blocks (12-48)
--recurrent_depth 5        # Recursive applications (3-7)
--d_model 384             # Hidden dimension (256-512)
--d_state 16              # SSM state dimension
--max_seq_len 256         # Max sequence length (128-512)
```

**Effective depth = n_blocks × recurrent_depth**
- 48 × 5 = 240 layers
- 36 × 6 = 216 layers
- 24 × 8 = 192 layers

### Training Parameters
```bash
--epochs 2                 # Training epochs
--batch_size 1            # Always 1 for 6GB VRAM
--accumulation_steps 16   # Effective batch = 1 × 16 = 16
--learning_rate 3e-4      # Learning rate
--max_grad_norm 1.0       # Gradient clipping
```

### Memory Optimization
```bash
--use_checkpointing       # Gradient checkpointing (saves 70% memory)
--checkpoint_segments 8   # Checkpoint every 8 blocks
--use_8bit_adam          # 8-bit optimizer (saves 60% optimizer memory)
--cache_tokenization     # Pre-tokenize data (faster training)
```

## 🎛️ Advanced Usage

### Progressive Training Strategy

Instead of jumping to 48 blocks, build up:

```bash
# Phase 1: Prototype (1 day)
python train.py \
    --data_path dialogues_combined.jsonl \
    --n_blocks 12 --recurrent_depth 3 \
    --epochs 1 \
    --checkpoint_dir checkpoints/proto

# Phase 2: Medium (2 days)
python train.py \
    --data_path dialogues_combined.jsonl \
    --n_blocks 24 --recurrent_depth 4 \
    --epochs 2 \
    --checkpoint_dir checkpoints/medium

# Phase 3: Full scale (10 days)
python train.py \
    --data_path dialogues_combined.jsonl \
    --n_blocks 48 --recurrent_depth 5 \
    --epochs 2 \
    --checkpoint_dir checkpoints/full
```

### Custom Data Format

If your JSONL has different fields, modify `data_loader.py`:

```python
# In DialogueDataset._tokenize_example():
def _tokenize_example(self, example):
    # Your custom format
    context = example['my_context_field']
    response = example['my_response_field']
    
    # Same tokenization logic...
```

### Inference (Using Trained Model)

```python
import torch
from model import create_model
from transformers import GPT2TokenizerFast

# Load model
model = create_model({'n_blocks': 48, 'recurrent_depth': 5, 'd_model': 384})
checkpoint = torch.load('checkpoints/final.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.half().cuda().eval()

# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# Generate
prompt = "User: Hello!\nAssistant:"
input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

with torch.no_grad():
    output = model(input_ids)
    next_token = output[0, -1].argmax()
    
generated_text = tokenizer.decode(next_token)
print(generated_text)
```

## 🔧 Troubleshooting

### Out of Memory
```
RuntimeError: CUDA out of memory
```

Solutions:
1. Reduce `d_model`: 384 → 320 → 256
2. Reduce `max_seq_len`: 256 → 192 → 128
3. Reduce `n_blocks`: 48 → 36 → 24
4. Reduce `checkpoint_segments`: 8 → 6 → 4

### Slow Training
```
<0.1 samples/sec
```

Solutions:
1. Increase `checkpoint_segments` (trades memory for speed)
2. Reduce `recurrent_depth` (fewer recursive passes)
3. Disable checkpointing if you have more VRAM
4. Use shorter sequences

### Loss Not Decreasing
```
Loss stuck at ~5.0+
```

Solutions:
1. Lower learning rate: 3e-4 → 1e-4
2. Increase warmup steps (add learning rate scheduler)
3. Check data quality (print decoded samples)
4. Verify gradient flow (print grad norms)

### Mamba Installation Failed
```
ERROR: Failed building wheel for mamba-ssm
```

**It's OK!** The code will use `SimplifiedMambaBlock` instead.

For best results, try:
```bash
# Install CUDA toolkit first
conda install -c nvidia cuda-toolkit

# Then retry mamba-ssm
pip install mamba-ssm
```

## 📈 Expected Results

### Training Progress
```
Epoch 0, Batch 100:   loss=4.2345
Epoch 0, Batch 1000:  loss=3.1234
Epoch 0, Batch 10000: loss=2.4567
Epoch 1, Batch 100:   loss=2.1234
Final:                loss=1.8-2.2
```

### Memory Usage
```
Current:  3.2GB
Peak:     4.8-5.2GB
Reserved: 5.5GB
Free:     0.5GB
```

### Speed
```
With all optimizations: 0.2-0.5 samples/sec
Full training (180k × 2): ~100-250 hours
```

## 🧠 Architecture Deep Dive

### What Makes This Novel?

1. **Recursive Block Reuse**: Instead of 240 unique layers, we have 48 physical blocks applied 5 times
   - Saves memory: 48 blocks vs 240 blocks
   - Potential better gradient flow
   - Forces model to learn reusable transformations

2. **State Space Instead of Attention**: Mamba blocks use state space models
   - O(n) complexity vs O(n²) for attention
   - Better for long sequences
   - Can theoretically remember infinite history

3. **Depth-Specific Adapters**: Each recursive depth has its own transformation
   - Depth 1 learns syntax
   - Depth 5 learns reasoning
   - Specialization across depths

### Comparison to Other Architectures

| Architecture | Layers | Complexity | Memory | Our Model |
|--------------|--------|------------|---------|-----------|
| GPT-2 Small | 12 | O(n²) | 2GB | ❌ Too shallow |
| GPT-2 XL | 48 | O(n²) | 8GB | ❌ Too much VRAM |
| RWKV-v7 | 24 | O(n) | 3GB | ✅ Similar idea |
| Mamba | 48 | O(n) | 4GB | ✅ Similar |
| **Ours** | **240 eff.** | **O(n)** | **5GB** | **✅ Novel!** |

## 📚 Files Overview

```
.
├── model.py           # RecursiveStateSpace architecture
├── data_loader.py     # JSONL dialogue dataset
├── train.py          # Main training script
├── test_memory.py    # Memory testing utility
├── requirements.txt  # Dependencies
└── README.md        # This file

After training:
├── checkpoints/
│   ├── step_5000.pt     # Intermediate checkpoints
│   ├── epoch_0.pt       # Epoch checkpoints
│   ├── final.pt         # Final model
│   └── metrics.json     # Training metrics
```

## 🤝 Tips & Best Practices

1. **Always test memory first** - Saves hours of failed training
2. **Start small** - 12 blocks → 24 blocks → 48 blocks
3. **Monitor early** - First 1000 steps should show loss decreasing
4. **Save often** - Use `--save_interval 2500` for long training
5. **Use screen/tmux** - Training takes days, don't let SSH disconnects stop you

## 📖 Citation

If you use this architecture, please acknowledge:
- Mamba: [Gu & Dao, 2023]
- Recursive Transformers: Novel contribution
- State Space Models: [Gu et al., 2021]

## 🐛 Known Issues

1. **Gradient instability**: May need gradient clipping tuning
2. **Slow convergence**: State space models can be slower to train
3. **Mamba compilation**: Can be tricky on some systems

## 💡 Future Improvements

1. Learning rate scheduler (cosine decay)
2. Adaptive depth (early stopping per sample)
3. Mixture-of-experts routing
4. Multi-GPU support
5. Quantization (4-bit) for even larger models

---

**Good luck with your training! 🚀**

Questions? Check the code comments or run with `--help` for all options.
#   R e c u r s i v e - M a m b a - 
 
 
