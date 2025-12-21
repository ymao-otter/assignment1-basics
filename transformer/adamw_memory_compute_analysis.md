# AdamW Memory and Compute Analysis

## Problem Setup

**Assumptions:**
- Using float32 for all tensors (4 bytes per element)
- d_ff = 4 × d_model
- Model hyperparameters: vocab_size, context_length, num_layers, d_model, num_heads
- batch_size (denoted as B)

---

## Part (a): Peak Memory Decomposition

### 1. Parameters

The model contains the following learnable parameters:

**Token Embeddings:**
```
vocab_size × d_model
```

**Per Transformer Block:**
- RMSNorm (ln1): d_model
- Multi-head attention: 4 × d_model² (Q, K, V, output projections)
- RMSNorm (ln2): d_model
- FFN: 3 × d_model × d_ff = 3 × d_model × 4d_model = 12d_model²

**Per block total:** d_model + 4d_model² + d_model + 12d_model² = 16d_model² + 2d_model

**All blocks:** num_layers × (16d_model² + 2d_model)

**Final RMSNorm:**
```
d_model
```

**LM Head:**
```
vocab_size × d_model
```

**Total Parameters:**
```
P = 2 × vocab_size × d_model + num_layers × (16d_model² + 2d_model) + d_model
```

Simplifying (dropping small terms):
```
P ≈ 2 × vocab_size × d_model + 16 × num_layers × d_model²
```

**Memory for Parameters (float32):**
```
M_params = 4P = 8 × vocab_size × d_model + 64 × num_layers × d_model² + O(d_model) bytes
```

---

### 2. Gradients

During training, we store gradients for all parameters. The gradient tensor has the same shape as the parameter tensor.

**Memory for Gradients:**
```
M_grads = 4P = 8 × vocab_size × d_model + 64 × num_layers × d_model² + O(d_model) bytes
```

---

### 3. Optimizer State (AdamW)

AdamW stores two state tensors per parameter:
- **exp_avg** (first moment): same shape as parameter
- **exp_avg_sq** (second moment): same shape as parameter

**Memory for Optimizer State:**
```
M_optimizer = 8P = 16 × vocab_size × d_model + 128 × num_layers × d_model² + O(d_model) bytes
```

---

### 4. Activations

We need to store intermediate activations for the backward pass. Let's denote:
- B = batch_size
- L = context_length
- H = num_heads
- D = d_model
- d_k = d_model / num_heads

**Per Transformer Block Activations:**

1. **Input to block:** B × L × D (stored for residual connection)

2. **RMSNorm output (ln1):** B × L × D

3. **Multi-head Attention:**
   - Q, K, V projections (before reshape): 3 × B × L × D
   - Q, K, V (after reshape): already counted above
   - Attention scores (Q^T K): B × H × L × L
   - Softmax output: B × H × L × L (can reuse memory from attention scores)
   - Attention output (before output proj): B × L × D
   - Output projection result: B × L × D

4. **After attention residual:** B × L × D (for second residual)

5. **RMSNorm output (ln2):** B × L × D

6. **Position-wise FFN:**
   - W1 output (before SiLU): B × L × d_ff = B × L × 4D
   - SiLU output: B × L × 4D (can potentially reuse memory)
   - W3 output: B × L × 4D
   - Element-wise product: B × L × 4D (can reuse memory)
   - W2 output: B × L × D

**Conservative estimate per block (storing all needed for backward):**
- Inputs/outputs for residuals: 3 × B × L × D
- RMSNorm outputs: 2 × B × L × D
- QKV projections: 3 × B × L × D
- Attention scores (after softmax): B × H × L × L = B × L × L × D
- Attention output: B × L × D
- FFN intermediate (W1, W3 outputs, product): 3 × B × L × 4D

**Per block activations:**
```
≈ 9BLD + BL²D + 12BLD = 21BLD + BL²D
```

**All transformer blocks:**
```
num_layers × (21BLD + BL²D)
```

**Final components:**
- Final RMSNorm output: B × L × D
- Logits (before cross-entropy): B × L × vocab_size
- Cross-entropy intermediate values: B × L × vocab_size (for softmax)

**Total Activations Memory (float32):**
```
M_activations = 4 × [num_layers × (21BLD + BL²D) + BLD + 2 × B × L × vocab_size]
              = 4 × [21 × num_layers × B × L × d_model + num_layers × B × L² × d_model 
                     + B × L × d_model + 2 × B × L × vocab_size]
```

Simplifying:
```
M_activations ≈ 84 × num_layers × B × L × d_model 
                + 4 × num_layers × B × L² × d_model
                + 8 × B × L × vocab_size bytes
```

---

### Total Peak Memory

```
M_total = M_params + M_grads + M_optimizer + M_activations
```

**Summary:**

| Component | Memory (bytes) |
|-----------|----------------|
| **Parameters** | 8 × vocab_size × d_model + 64 × num_layers × d_model² |
| **Gradients** | 8 × vocab_size × d_model + 64 × num_layers × d_model² |
| **Optimizer State** | 16 × vocab_size × d_model + 128 × num_layers × d_model² |
| **Activations** | 84 × num_layers × B × L × d_model + 4 × num_layers × B × L² × d_model + 8 × B × L × vocab_size |
| **TOTAL** | 32 × vocab_size × d_model + 256 × num_layers × d_model² + 84 × num_layers × B × L × d_model + 4 × num_layers × B × L² × d_model + 8 × B × L × vocab_size |

Where L = context_length

---

## Part (b): GPT-2 XL Memory and Maximum Batch Size

**GPT-2 XL Configuration:**
- vocab_size = 50,257
- context_length = 1,024
- num_layers = 48
- d_model = 1,600
- num_heads = 25
- d_ff = 6,400

### Calculating Constants

**Parameters:**
```
P = 2 × 50,257 × 1,600 + 48 × 16 × 1,600²
  = 160,822,400 + 48 × 16 × 2,560,000
  = 160,822,400 + 1,966,080,000
  = 2,126,902,400
```

**Fixed Memory (independent of batch_size):**

Parameters + Gradients + Optimizer State = 12P
```
M_fixed = 12 × 4 × 2,126,902,400
        = 48 × 2,126,902,400
        = 102,091,315,200 bytes
        ≈ 95.09 GB
```

Wait, this doesn't look right. Let me recalculate more carefully.

**Parameters (float32):**
```
M_params = 4 × 2,126,902,400 = 8,507,609,600 bytes ≈ 7.92 GB
```

**Gradients (float32):**
```
M_grads = 4 × 2,126,902,400 = 8,507,609,600 bytes ≈ 7.92 GB
```

**Optimizer State (2 × parameters for AdamW):**
```
M_optimizer = 8 × 2,126,902,400 = 17,015,219,200 bytes ≈ 15.84 GB
```

**Fixed memory total:**
```
M_fixed = 8,507,609,600 + 8,507,609,600 + 17,015,219,200
        = 34,030,438,400 bytes
        ≈ 31.69 GB
```

**Variable Memory (depends on batch_size = B):**

With L = 1,024, num_layers = 48, d_model = 1,600, vocab_size = 50,257:

```
M_activations = 84 × 48 × B × 1,024 × 1,600 
                + 4 × 48 × B × 1,024² × 1,600
                + 8 × B × 1,024 × 50,257

              = 6,559,948,800 × B 
                + 201,326,592 × B
                + 411,704,320 × B

              = (6,559,948,800 + 201,326,592 + 411,704,320) × B
              = 7,172,979,712 × B bytes
              ≈ 6.68 × B GB
```

**Total Memory Expression:**
```
M_total = 34,030,438,400 + 7,172,979,712 × B bytes
```

Converting to GB (using 1 GB = 1,073,741,824 bytes):
```
M_total ≈ 31.69 + 6.68 × B GB
```

**Expression in form a × batch_size + b:**
```
M_total = 6.68 × B + 31.69 GB
```

Or more precisely:
```
a = 7,172,979,712 bytes ≈ 6.68 GB
b = 34,030,438,400 bytes ≈ 31.69 GB
```

**Maximum Batch Size for 80 GB:**

```
6.68 × B + 31.69 ≤ 80
6.68 × B ≤ 48.31
B ≤ 7.23
```

**Maximum batch_size = 7**

---

## Part (c): FLOPs for One AdamW Step

One training step includes:
1. Forward pass
2. Backward pass (typically 2× forward pass FLOPs)
3. Optimizer update

### Forward Pass FLOPs

For matrix multiply C = A @ B where A is (m, k) and B is (k, n):
```
FLOPs = 2mkn
```

**Per Transformer Block (with batch B, sequence L):**

1. **Attention Projections (Q, K, V, output):**
   - Each: 2 × B × L × d_model × d_model
   - Total: 4 × 2BLd_model²

2. **Attention Computation:**
   - Q^T K: 2 × B × H × L × d_k × L = 2BHL²d_k = 2BL²d_model
   - Softmax @ V: 2 × B × H × L × L × d_k = 2BHL²d_k = 2BL²d_model
   - Total: 4BL²d_model

3. **FFN:**
   - W1: 2 × B × L × d_model × d_ff = 2BL × d_model × 4d_model = 8BLd_model²
   - W2: 2 × B × L × d_ff × d_model = 8BLd_model²
   - W3: 2 × B × L × d_model × d_ff = 8BLd_model²
   - Total: 24BLd_model²

**Per block total:**
```
F_block = 8BLd_model² + 4BL²d_model + 24BLd_model² = 32BLd_model² + 4BL²d_model
```

**All blocks:**
```
F_blocks = num_layers × (32BLd_model² + 4BL²d_model)
```

**LM Head:**
```
F_lm_head = 2 × B × L × d_model × vocab_size
```

**Token Embeddings:** 0 FLOPs (lookup operation)

**Total Forward Pass FLOPs:**
```
F_forward = 32 × num_layers × B × L × d_model² 
            + 4 × num_layers × B × L² × d_model
            + 2 × B × L × d_model × vocab_size
```

### Backward Pass FLOPs

The backward pass requires computing gradients through all operations. For matrix multiplications, the backward pass requires:
- Gradient w.r.t. input: 2× forward FLOPs
- Gradient w.r.t. weights: counted separately but approximately equal to forward

**Rule of thumb:** Backward pass ≈ 2 × Forward pass FLOPs

```
F_backward ≈ 2 × F_forward
```

### Optimizer Update FLOPs

AdamW update for each parameter:
1. Update first moment: 2 operations per parameter (multiply + add)
2. Update second moment: 3 operations per parameter (square + multiply + add)
3. Compute update: ~5 operations per parameter (bias correction, division, sqrt, multiply, add)

**Total: ~10 operations per parameter**

```
F_optimizer = 10 × P
```

where P is the total number of parameters.

This is negligible compared to forward/backward passes since:
```
F_optimizer ≈ 10P
F_forward ≈ 32 × num_layers × B × L × d_model² ≈ 2BL × 16 × num_layers × d_model²
```

For typical values (B=1024, L=1024, num_layers=48, d_model=1600), forward pass is ~millions of times larger.

**Total FLOPs per AdamW Step:**
```
F_total ≈ F_forward + F_backward + F_optimizer
        ≈ 3 × F_forward
        = 3 × [32 × num_layers × B × L × d_model² 
               + 4 × num_layers × B × L² × d_model
               + 2 × B × L × d_model × vocab_size]
```

**Simplified (dominant terms):**
```
F_total ≈ 96 × num_layers × B × L × d_model² + 12 × num_layers × B × L² × d_model + 6 × B × L × d_model × vocab_size
```

---

## Part (d): Training Time for GPT-2 XL

**Given:**
- NVIDIA A100: 19.5 TFLOP/s peak (float32)
- MFU (Model FLOPs Utilization): 50%
- Training: 400K steps
- batch_size = 1,024
- GPT-2 XL configuration (as above)

### Step 1: Calculate FLOPs per Training Step

Using the formula from part (c) with:
- B = 1,024
- L = 1,024
- num_layers = 48
- d_model = 1,600
- vocab_size = 50,257

**Forward pass FLOPs:**
```
F_forward = 32 × 48 × 1,024 × 1,024 × 1,600²
            + 4 × 48 × 1,024 × 1,024² × 1,600
            + 2 × 1,024 × 1,024 × 1,600 × 50,257

          = 32 × 48 × 1,048,576 × 2,560,000
            + 4 × 48 × 1,048,576 × 1,024 × 1,600
            + 2 × 1,048,576 × 1,600 × 50,257

          = 4,127,195,136,000,000
            + 322,122,547,200,000
            + 168,631,018,496,000

          = 4,617,948,701,696,000 FLOPs
          ≈ 4.62 × 10^15 FLOPs
          ≈ 4,618 TFLOP
```

**Total per step (including backward at 2×):**
```
F_step = 3 × F_forward ≈ 3 × 4.62 × 10^15 = 1.386 × 10^16 FLOPs
       ≈ 13,854 TFLOP
```

### Step 2: Calculate Effective Throughput

**Theoretical peak:** 19.5 TFLOP/s
**Effective (50% MFU):** 0.5 × 19.5 = 9.75 TFLOP/s

### Step 3: Calculate Time per Step

```
Time per step = FLOPs per step / Effective throughput
              = 1.386 × 10^16 / (9.75 × 10^12)
              = 1,421.5 seconds
              ≈ 23.69 minutes
```

### Step 4: Calculate Total Training Time

```
Total time = 400,000 steps × 1,421.5 seconds/step
           = 568,615,385 seconds
           = 9,476,923 minutes
           = 157,949 hours
           = 6,581 days
```

**Training would take approximately 6,581 days (18.0 years).**

### Justification

The calculation follows these steps:
1. **FLOPs per forward pass**: Computed from matrix multiplications in transformer blocks and LM head
2. **Total FLOPs per step**: 3× forward pass (1× forward + 2× backward)
3. **Effective throughput**: 50% MFU × 19.5 TFLOP/s = 9.75 TFLOP/s
4. **Time = FLOPs / Throughput**: Standard computation time formula
5. **Total time**: 400K steps × time per step

The result shows that training GPT-2 XL on a single A100 GPU with this batch size would take ~18 years, which is why distributed training across many GPUs is essential for large model training.

---

## Summary

### Part (a) - Peak Memory:

| Component | Memory Expression |
|-----------|-------------------|
| Parameters | 8 × vocab_size × d_model + 64 × num_layers × d_model² bytes |
| Gradients | 8 × vocab_size × d_model + 64 × num_layers × d_model² bytes |
| Optimizer State | 16 × vocab_size × d_model + 128 × num_layers × d_model² bytes |
| Activations | 84 × num_layers × B × L × d_model + 4 × num_layers × B × L² × d_model + 8 × B × L × vocab_size bytes |
| **TOTAL** | **32 × vocab_size × d_model + 256 × num_layers × d_model² + (84 × num_layers × L × d_model + 4 × num_layers × L² × d_model + 8 × L × vocab_size) × B bytes** |

### Part (b) - GPT-2 XL:

**Memory expression:**
```
M_total = 6.68 × batch_size + 31.69 GB
```

More precisely:
```
a = 7,172,979,712 bytes ≈ 6.68 GB
b = 34,030,438,400 bytes ≈ 31.69 GB
```

**Maximum batch size for 80 GB memory: 7**

### Part (c) - FLOPs per AdamW Step:

```
F_total = 3 × [32 × num_layers × batch_size × context_length × d_model² 
               + 4 × num_layers × batch_size × context_length² × d_model
               + 2 × batch_size × context_length × d_model × vocab_size]
```

The factor of 3 comes from: 1× forward pass + 2× backward pass.

### Part (d) - Training Time:

**Training GPT-2 XL (batch_size=1024, 400K steps) on single A100 at 50% MFU:**

**≈ 6,581 days (18.0 years)**

**Justification:** Each step requires ~13,854 TFLOP. At 50% MFU, the A100 provides 9.75 TFLOP/s effective throughput, giving ~1,422 seconds per step. For 400K steps: 400,000 × 1,422 / 86,400 ≈ 6,581 days.

