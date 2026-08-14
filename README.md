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
