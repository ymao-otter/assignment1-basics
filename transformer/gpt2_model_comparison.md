# GPT-2 Model Size Comparison

## Model Configurations

| Model | Layers | d_model | Heads | d_ff | vocab_size | context_length |
|-------|--------|---------|-------|------|------------|----------------|
| **Small** | 12 | 768 | 12 | 3,072 | 50,257 | 1,024 |
| **Medium** | 24 | 1,024 | 16 | 4,096 | 50,257 | 1,024 |
| **Large** | 36 | 1,280 | 20 | 5,120 | 50,257 | 1,024 |
| **XL** | 48 | 1,600 | 25 | 6,400 | 50,257 | 1,024 |

Note: d_ff = 4 × d_model (standard GPT-2 ratio)

---

## Parameter Count Analysis

### Formula Summary

For each model, total parameters consist of:

1. **Token Embeddings**: vocab_size × d_model
2. **Transformer Blocks** (×num_layers):
   - ln1: d_model
   - Attention (4 projections): 4 × (d_model × d_model)
   - ln2: d_model
   - FFN (3 projections): d_model × d_ff + d_ff × d_model + d_model × d_ff = 3 × d_model × d_ff
3. **Final LayerNorm**: d_model
4. **LM Head**: vocab_size × d_model

### GPT-2 Small (12L, 768D, 12H)

```
Token Embeddings = 50,257 × 768 = 38,597,376

Per Block:
  ln1 = 768
  Attention = 4 × (768 × 768) = 2,359,296
  ln2 = 768
  FFN = 3 × (768 × 3,072) = 7,077,888
  Total per block = 768 + 2,359,296 + 768 + 7,077,888 = 9,438,720

All Blocks = 12 × 9,438,720 = 113,264,640

Final LayerNorm = 768

LM Head = 50,257 × 768 = 38,597,376

Total = 38,597,376 + 113,264,640 + 768 + 38,597,376
      = 190,460,160 parameters
```

**GPT-2 Small: ~190.5M parameters**

### GPT-2 Medium (24L, 1024D, 16H)

```
Token Embeddings = 50,257 × 1,024 = 51,463,168

Per Block:
  ln1 = 1,024
  Attention = 4 × (1,024 × 1,024) = 4,194,304
  ln2 = 1,024
  FFN = 3 × (1,024 × 4,096) = 12,582,912
  Total per block = 1,024 + 4,194,304 + 1,024 + 12,582,912 = 16,779,264

All Blocks = 24 × 16,779,264 = 402,702,336

Final LayerNorm = 1,024

LM Head = 50,257 × 1,024 = 51,463,168

Total = 51,463,168 + 402,702,336 + 1,024 + 51,463,168
      = 505,629,696 parameters
```

**GPT-2 Medium: ~505.6M parameters**

### GPT-2 Large (36L, 1280D, 20H)

```
Token Embeddings = 50,257 × 1,280 = 64,328,960

Per Block:
  ln1 = 1,280
  Attention = 4 × (1,280 × 1,280) = 6,553,600
  ln2 = 1,280
  FFN = 3 × (1,280 × 5,120) = 19,660,800
  Total per block = 1,280 + 6,553,600 + 1,280 + 19,660,800 = 26,216,960

All Blocks = 36 × 26,216,960 = 943,810,560

Final LayerNorm = 1,280

LM Head = 50,257 × 1,280 = 64,328,960

Total = 64,328,960 + 943,810,560 + 1,280 + 64,328,960
      = 1,072,469,760 parameters
```

**GPT-2 Large: ~1.07B parameters**

### GPT-2 XL (48L, 1600D, 25H)

```
Token Embeddings = 50,257 × 1,600 = 80,411,200

Per Block:
  ln1 = 1,600
  Attention = 4 × (1,600 × 1,600) = 10,240,000
  ln2 = 1,600
  FFN = 3 × (1,600 × 6,400) = 30,720,000
  Total per block = 1,600 + 10,240,000 + 1,600 + 30,720,000 = 40,963,200

All Blocks = 48 × 40,963,200 = 1,966,233,600

Final LayerNorm = 1,600

LM Head = 50,257 × 1,600 = 80,411,200

Total = 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200
      = 2,127,057,600 parameters
```

**GPT-2 XL: ~2.13B parameters**

### Parameter Summary Table

| Model | Total Params | Token Emb | Transformer Blocks | LM Head | Memory (FP32) |
|-------|--------------|-----------|-------------------|---------|---------------|
| **Small** | 190,460,160 | 38,597,376 (20.3%) | 113,264,640 (59.5%) | 38,597,376 (20.3%) | ~0.76 GB |
| **Medium** | 505,629,696 | 51,463,168 (10.2%) | 402,702,336 (79.6%) | 51,463,168 (10.2%) | ~2.02 GB |
| **Large** | 1,072,469,760 | 64,328,960 (6.0%) | 943,810,560 (88.0%) | 64,328,960 (6.0%) | ~4.29 GB |
| **XL** | 2,127,057,600 | 80,411,200 (3.8%) | 1,966,233,600 (92.4%) | 80,411,200 (3.8%) | ~7.92 GB |

**Observation**: As model size increases, the transformer blocks dominate increasingly (59.5% → 92.4%), while the embedding/head layers become proportionally smaller.

---

## FLOPs Analysis

### FLOP Calculation Setup

For input sequence length = 1,024 tokens, batch size = 1.

Matrix multiply FLOPs for C = A @ B where A is (m, k) and B is (k, n):
```
FLOPs = 2 × m × k × n
```

### Per-Block FLOPs Formulas

Let:
- L = context_length = 1,024
- D = d_model
- H = num_heads
- d_k = D / H (dimension per head)
- F = d_ff

**Attention Projections:**
- Q projection: 2 × L × D × D
- K projection: 2 × L × D × D
- V projection: 2 × L × D × D
- Output projection: 2 × L × D × D
- **Total**: 8 × L × D²

**Attention Computation:**
- QK^T: H × (2 × L × d_k × L) = 2 × L² × D
- Attn@V: H × (2 × L × L × d_k) = 2 × L² × D
- **Total**: 4 × L² × D

**FFN:**
- w1: 2 × L × D × F
- w2: 2 × L × F × D
- w3: 2 × L × D × F
- **Total**: 6 × L × D × F = 6 × L × D × (4D) = 24 × L × D²

**Total per block**: 8LD² + 4L²D + 24LD² = 32LD² + 4L²D

**LM Head**: 2 × L × D × vocab_size

### GPT-2 Small (12L, 768D)

```
D = 768, L = 1,024, num_layers = 12

Per Block:
  Attention Projections = 8 × 1,024 × 768² = 4,831,838,208
  Attention Computation = 4 × 1,024² × 768 = 3,221,225,472
  FFN = 24 × 1,024 × 768² = 14,495,514,624
  Total per block = 22,548,578,304

All Blocks = 12 × 22,548,578,304 = 270,582,939,648

LM Head = 2 × 1,024 × 768 × 50,257 = 78,999,027,712

Total FLOPs = 270,582,939,648 + 78,999,027,712
            = 349,581,967,360
```

**GPT-2 Small: ~349.6B FLOPs**

Breaking down per block:
- Attention Projections: 4.83B (21.4%)
- Attention Computation: 3.22B (14.3%)
- FFN: 14.50B (64.3%)

### GPT-2 Medium (24L, 1024D)

```
D = 1,024, L = 1,024, num_layers = 24

Per Block:
  Attention Projections = 8 × 1,024 × 1,024² = 8,589,934,592
  Attention Computation = 4 × 1,024² × 1,024 = 4,294,967,296
  FFN = 24 × 1,024 × 1,024² = 25,769,803,776
  Total per block = 38,654,705,664

All Blocks = 24 × 38,654,705,664 = 927,712,935,936

LM Head = 2 × 1,024 × 1,024 × 50,257 = 105,326,493,696

Total FLOPs = 927,712,935,936 + 105,326,493,696
            = 1,033,039,429,632
```

**GPT-2 Medium: ~1.03T FLOPs**

Breaking down per block:
- Attention Projections: 8.59B (22.2%)
- Attention Computation: 4.29B (11.1%)
- FFN: 25.77B (66.7%)

### GPT-2 Large (36L, 1280D)

```
D = 1,280, L = 1,024, num_layers = 36

Per Block:
  Attention Projections = 8 × 1,024 × 1,280² = 13,421,772,800
  Attention Computation = 4 × 1,024² × 1,280 = 5,368,709,120
  FFN = 24 × 1,024 × 1,280² = 40,265,318,400
  Total per block = 59,055,800,320

All Blocks = 36 × 59,055,800,320 = 2,126,008,811,520

LM Head = 2 × 1,024 × 1,280 × 50,257 = 131,733,913,600

Total FLOPs = 2,126,008,811,520 + 131,733,913,600
            = 2,257,742,725,120
```

**GPT-2 Large: ~2.26T FLOPs**

Breaking down per block:
- Attention Projections: 13.42B (22.7%)
- Attention Computation: 5.37B (9.1%)
- FFN: 40.27B (68.2%)

### GPT-2 XL (48L, 1600D)

```
D = 1,600, L = 1,024, num_layers = 48

Per Block:
  Attention Projections = 8 × 1,024 × 1,600² = 20,971,520,000
  Attention Computation = 4 × 1,024² × 1,600 = 6,710,886,400
  FFN = 24 × 1,024 × 1,600² = 62,914,560,000
  Total per block = 90,596,966,400

All Blocks = 48 × 90,596,966,400 = 4,348,654,387,200

LM Head = 2 × 1,024 × 1,600 × 50,257 = 164,679,229,440

Total FLOPs = 4,348,654,387,200 + 164,679,229,440
            = 4,513,333,616,640
```

**GPT-2 XL: ~4.51T FLOPs**

Breaking down per block:
- Attention Projections: 20.97B (23.1%)
- Attention Computation: 6.71B (7.4%)
- FFN: 62.91B (69.4%)

### FLOPs Summary Table

| Model | Total FLOPs | Transformer Blocks | LM Head | Blocks % |
|-------|-------------|-------------------|---------|----------|
| **Small** | 349.6B | 270.6B | 79.0B | 77.4% |
| **Medium** | 1,033.0B | 927.7B | 105.3B | 89.8% |
| **Large** | 2,257.7B | 2,126.0B | 131.7B | 94.2% |
| **XL** | 4,513.3B | 4,348.7B | 164.7B | 96.4% |

---

## Scaling Analysis

### Component Breakdown Within Transformer Blocks

| Model | Attn Proj % | Attn Compute % | FFN % |
|-------|-------------|----------------|-------|
| **Small** | 21.4% | 14.3% | 64.3% |
| **Medium** | 22.2% | 11.1% | 66.7% |
| **Large** | 22.7% | 9.1% | 68.2% |
| **XL** | 23.1% | 7.4% | 69.4% |

### Key Observations

#### 1. **FFN Dominates and Grows**
The Feed-Forward Network (FFN) consistently takes up the most FLOPs within each block, **increasing from 64.3% to 69.4%** as models grow.

**Why?** FFN FLOPs scale as O(LD²) where the hidden dimension grows (d_ff = 4 × d_model), resulting in quadratic scaling with d_model.

#### 2. **Attention Computation Shrinks Proportionally**
Attention computation (QK^T and Attn@V) **decreases from 14.3% to 7.4%** as models grow.

**Why?** Attention computation scales as O(L²D), which is linear in d_model but quadratic in sequence length. Since sequence length is fixed at 1,024, this component grows slower than FFN as d_model increases.

#### 3. **Attention Projections Stay Relatively Constant**
Attention projections (Q, K, V, Output) remain relatively stable at **~21-23%**.

**Why?** These scale as O(LD²), the same as FFN, but there are 4 such projections vs. 3 FFN layers, and the FFN layers use larger intermediate dimensions (4D vs D).

#### 4. **LM Head Becomes Less Significant**
The LM head's contribution **decreases from 22.6% to 3.6%** of total FLOPs.

**Why?** LM head FLOPs scale as O(L × D × vocab_size). Since vocab_size is constant, this scales linearly with d_model, while transformer block FLOPs scale quadratically (O(LD²)).

#### 5. **Transformer Blocks Dominate Overall**
Transformer blocks grow from **77.4% to 96.4%** of total FLOPs as models scale.

**Why?** Combining observations #1 and #4: block FLOPs scale quadratically while LM head scales linearly.

### Mathematical Scaling

Given the formulas:
- **Transformer Blocks**: num_layers × (32LD² + 4L²D)
- **LM Head**: 2 × L × D × vocab_size

As d_model increases with fixed L and vocab_size:
- Blocks grow as O(num_layers × D²)
- LM Head grows as O(D)

The **ratio** of transformer blocks to LM head:
```
Ratio = (num_layers × (32LD² + 4L²D)) / (2 × L × D × vocab_size)
      ≈ (num_layers × 32D) / (2 × vocab_size)   [when LD >> L²]
      = (16 × num_layers × D) / vocab_size
```

This ratio increases linearly with both num_layers and d_model, explaining why larger models are increasingly dominated by transformer block computation.

### Practical Implications

1. **Optimization Focus**: For large models, optimizing FFN operations (e.g., fused kernels, quantization) yields the most impact.

2. **Sequence Length Scaling**: If we increased context_length, attention computation (O(L²D)) would grow quadratically and become more significant, while linear projections (O(LD²)) would grow linearly.

3. **Model Parallelism**: The FFN layers are embarrassingly parallel across the d_ff dimension, making them good candidates for tensor parallelism.

4. **Flash Attention**: While flash attention optimizes memory and improves speed, it affects a decreasing proportion of total FLOPs as models grow (14.3% → 7.4%).

5. **Vocabulary Size**: Increasing vocabulary has minimal impact on large models (since LM head is already small), but matters more for small models.

---

## Visual Scaling Summary

### FLOP Distribution by Component (% of Total)

```
Component             | Small  | Medium | Large  | XL
---------------------|--------|--------|--------|--------
Transformer Blocks   | 77.4%  | 89.8%  | 94.2%  | 96.4%  ↑
  └─ FFN            | 49.8%  | 59.9%  | 64.2%  | 67.0%  ↑↑
  └─ Attn Proj      | 16.5%  | 20.0%  | 21.4%  | 22.3%  ↑
  └─ Attn Compute   | 11.1%  | 10.0%  |  8.6%  |  7.1%  ↓↓
LM Head              | 22.6%  | 10.2%  |  5.8%  |  3.6%  ↓↓
```

**Legend:**
- ↑↑ = Strongly increasing proportion
- ↑ = Moderately increasing proportion  
- ↓ = Moderately decreasing proportion
- ↓↓ = Strongly decreasing proportion

### Parameter Distribution by Component (% of Total)

```
Component             | Small  | Medium | Large  | XL
---------------------|--------|--------|--------|--------
Transformer Blocks   | 59.5%  | 79.6%  | 88.0%  | 92.4%  ↑↑
Token Embeddings     | 20.3%  | 10.2%  |  6.0%  |  3.8%  ↓↓
LM Head              | 20.3%  | 10.2%  |  6.0%  |  3.8%  ↓↓
```

### Absolute Scaling

| Model | Params Ratio | FLOPs Ratio | Size vs Small |
|-------|--------------|-------------|---------------|
| **Small** | 1.00× | 1.00× | Baseline |
| **Medium** | 2.65× | 2.95× | ~3× |
| **Large** | 5.63× | 6.46× | ~6× |
| **XL** | 11.17× | 12.91× | ~13× |

**Key Insight**: FLOPs scale slightly faster than parameters because larger models use proportionally more matrix multiplications relative to their parameter count.


