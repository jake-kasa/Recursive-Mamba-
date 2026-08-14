# RecursiveStateSpace: 48-Block Mamba Language Model

Training a novel hybrid state space model on dialogue data, optimized for RTX 2060 (6GB VRAM).

## 🏗️ Architecture

**RecursiveStateSpace** combines:
- **48 physical Mamba (state space) blocks**
- **5 recursive depths** = 240 effective layers!
- **Memory-efficient**: Only stores 48 blocks, applies them recursively
- **Linear complexity**: $\mathcal{O}(n)$ vs $\mathcal{O}(n^2)$ for transformers

### Why This Works

Traditional 240-layer transformer: **Impossible on 6GB**  
Our approach: **48 blocks × 5 recursions = Same depth, manageable memory**

Each recursive pass refines understanding:
- **Depth 1–2**: Syntax, grammar, basic patterns
- **Depth 3–4**: Semantics, relationships, context
- **Depth 5**: Abstract reasoning, coherence, world knowledge

---

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


# Test with default config (48 blocks, depth 5, d=384)
python dataset_alpaca.py

# Test multiple configurations
python dataset_alpaca.py --test_all

# Test custom config
python dataset_alpaca.py --n_blocks 36 --d_model 320

✅ SUCCESS! Model fits in memory
   Peak usage: 4.82GB / 6.0GB
   Headroom: 1.18GB

### Data prep
{"context": ["User: Hello", "Assistant: Hi!"], "response": "How can I help?", "source": "lmsys"}

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

python train.py \
    --data_path dialogues_combined.jsonl \
    --epochs 2 \
    --n_blocks 24 \
    --recurrent_depth 5 \
    --d_model 384
    
# Estimated time: ~100 hours (4 days)




