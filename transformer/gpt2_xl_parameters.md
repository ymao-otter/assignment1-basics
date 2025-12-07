# GPT-2 XL Parameter Calculation

## Configuration

| Parameter | Value |
|-----------|-------|
| vocab_size | 50,257 |
| context_length | 1,024 |
| num_layers | 48 |
| d_model | 1,600 |
| num_heads | 25 |
| d_ff | 6,400 |

## Model Architecture

Based on the `TransformerLM` implementation, the model consists of:

1. **Token Embeddings** (`Embedding`)
2. **Transformer Blocks** (×48)
3. **Final Layer Norm** (`RMSNorm`)
4. **LM Head** (`Linear`)

### Note on RoPE
RoPE (Rotary Position Embedding) uses registered buffers, not learnable parameters, so it contributes 0 trainable parameters.

## Detailed Parameter Count

### 1. Token Embeddings

The token embeddings map from vocabulary to model dimension:

```
Parameters = vocab_size × d_model
           = 50,257 × 1,600
           = 80,411,200
```

### 2. Each Transformer Block

Each `TransformerBlock` contains:
- **ln1** (RMSNorm): weight vector
- **attn** (MultiheadSelfAttention): Q, K, V projections and output projection
- **ln2** (RMSNorm): weight vector
- **ffn** (FFN/SwiGLU): three linear layers (w1, w2, w3)

#### 2.1 First Layer Norm (ln1)

```
Parameters = d_model
           = 1,600
```

#### 2.2 Multi-head Self Attention (attn)

The attention module has 4 linear layers (no bias):
- **q_proj**: d_model → d_model
- **k_proj**: d_model → d_model
- **v_proj**: d_model → d_model
- **output_proj**: d_model → d_model

```
Parameters per layer = out_features × in_features
                     = d_model × d_model
                     = 1,600 × 1,600
                     = 2,560,000

Total attention parameters = 4 × 2,560,000
                           = 10,240,000
```

#### 2.3 Second Layer Norm (ln2)

```
Parameters = d_model
           = 1,600
```

#### 2.4 Feed-Forward Network (ffn)

The SwiGLU FFN has 3 linear layers (no bias):
- **w1**: d_model → d_ff (up-projection)
- **w2**: d_ff → d_model (down-projection)
- **w3**: d_model → d_ff (gate projection)

```
w1 parameters = d_ff × d_model
              = 6,400 × 1,600
              = 10,240,000

w2 parameters = d_model × d_ff
              = 1,600 × 6,400
              = 10,240,000

w3 parameters = d_ff × d_model
              = 6,400 × 1,600
              = 10,240,000

Total FFN parameters = 10,240,000 + 10,240,000 + 10,240,000
                     = 30,720,000
```

#### 2.5 Total per Transformer Block

```
Parameters per block = ln1 + attn + ln2 + ffn
                     = 1,600 + 10,240,000 + 1,600 + 30,720,000
                     = 40,963,200
```

#### 2.6 All Transformer Blocks

```
Total parameters = num_layers × parameters_per_block
                 = 48 × 40,963,200
                 = 1,966,233,600
```

### 3. Final Layer Norm (ln_final)

```
Parameters = d_model
           = 1,600
```

### 4. LM Head

The language modeling head projects from model dimension to vocabulary:

```
Parameters = vocab_size × d_model
           = 50,257 × 1,600
           = 80,411,200
```

## Total Trainable Parameters

```
Total = token_embeddings + all_transformer_blocks + ln_final + lm_head
      = 80,411,200 + 1,966,233,600 + 1,600 + 80,411,200
      = 2,127,057,600 parameters
```

**Total: 2,127,057,600 parameters (~2.13 billion parameters)**

## Memory Requirements

Each parameter is stored as a single-precision (32-bit) floating point number.

```
Bytes per parameter = 32 bits / 8 bits per byte
                    = 4 bytes
```

```
Total memory = total_parameters × bytes_per_parameter
             = 2,127,057,600 × 4
             = 8,508,230,400 bytes
```

Converting to more readable units:

```
= 8,508,230,400 / 1024 KB
= 8,309,803.125 KB

= 8,309,803.125 / 1024 MB
= 8,114.0655 MB

= 8,114.0655 / 1024 GB
≈ 7.924 GB
```

**Memory required to load model: ~7.92 GB** (using binary units: 1 KB = 1024 bytes)

or

**~8.51 GB** (using decimal units: 1 KB = 1000 bytes)

### Note

This calculation only accounts for the model parameters themselves. In practice, additional memory is required for:
- Gradients (same size as parameters during training)
- Optimizer states (e.g., AdamW stores first and second moments, ~2× parameter memory)
- Activations during forward/backward passes
- Input/output buffers

For training with AdamW optimizer, you would need approximately:
- Parameters: ~7.92 GB
- Gradients: ~7.92 GB
- Optimizer states: ~15.84 GB
- **Total for training: ~31.68 GB** (plus activations)

---

## FLOPs Analysis for Forward Pass

### Setup

We analyze the floating-point operations (FLOPs) required for a forward pass with:
- Input sequence length: **context_length = 1,024 tokens**
- Batch size: **1** (can scale linearly for larger batches)
- d_k (dimension per head): d_model / num_heads = 1,600 / 25 = **64**

### FLOP Counting Convention

For a matrix multiply `C = A @ B` where:
- A has shape (m, k)
- B has shape (k, n)
- C has shape (m, n)

The FLOPs required are:
```
FLOPs = 2 × m × k × n
```

This accounts for k multiplications and k additions (approximated as 2k operations) for each of the m×n output elements.

### Matrix Multiplies in the Model

#### 1. Token Embeddings

Token embeddings use **lookup indexing**, not matrix multiplication.
```
FLOPs = 0
```

#### 2. Each Transformer Block

Each of the 48 transformer blocks contains the following matrix multiplies:

##### 2.1 Attention Projections

**Q Projection:**
- Input shape: (1, 1024, 1600)
- Weight shape: (1600, 1600)
- Output shape: (1, 1024, 1600)

```
FLOPs = 2 × 1,024 × 1,600 × 1,600
      = 5,242,880,000
```

**K Projection:**
- Same dimensions as Q projection

```
FLOPs = 2 × 1,024 × 1,600 × 1,600
      = 5,242,880,000
```

**V Projection:**
- Same dimensions as Q projection

```
FLOPs = 2 × 1,024 × 1,600 × 1,600
      = 5,242,880,000
```

##### 2.2 Attention Score Computation (QK^T)

After projections, Q and K are reshaped to (1, 25, 1024, 64) for multi-head attention.

For each head, we compute: (1024, 64) @ (64, 1024) → (1024, 1024)

```
FLOPs per head = 2 × 1,024 × 64 × 1,024
               = 134,217,728

Total for all heads = 25 × 134,217,728
                    = 3,355,443,200
```

##### 2.3 Attention Output Computation (Softmax(QK^T) @ V)

Shape: (1, 25, 1024, 1024) @ (1, 25, 1024, 64) → (1, 25, 1024, 64)

For each head: (1024, 1024) @ (1024, 64) → (1024, 64)

```
FLOPs per head = 2 × 1,024 × 1,024 × 64
               = 134,217,728

Total for all heads = 25 × 134,217,728
                    = 3,355,443,200
```

##### 2.4 Attention Output Projection

After concatenating heads back to (1, 1024, 1600):

```
FLOPs = 2 × 1,024 × 1,600 × 1,600
      = 5,242,880,000
```

##### 2.5 FFN Layer w1 (SwiGLU gate)

- Input: (1, 1024, 1600)
- Weight: (1600, 6400)
- Output: (1, 1024, 6400)

```
FLOPs = 2 × 1,024 × 1,600 × 6,400
      = 20,971,520,000
```

##### 2.6 FFN Layer w2 (down-projection)

- Input: (1, 1024, 6400)
- Weight: (6400, 1600)
- Output: (1, 1024, 1600)

```
FLOPs = 2 × 1,024 × 6,400 × 1,600
      = 20,971,520,000
```

##### 2.7 FFN Layer w3 (SwiGLU value)

- Input: (1, 1024, 1600)
- Weight: (1600, 6400)
- Output: (1, 1024, 6400)

```
FLOPs = 2 × 1,024 × 1,600 × 6,400
      = 20,971,520,000
```

##### 2.8 Total FLOPs per Transformer Block

```
FLOPs per block = Q_proj + K_proj + V_proj + QK^T + Attn@V + Output_proj + w1 + w2 + w3
                = 5,242,880,000 + 5,242,880,000 + 5,242,880,000 + 3,355,443,200 
                  + 3,355,443,200 + 5,242,880,000 + 20,971,520,000 + 20,971,520,000 
                  + 20,971,520,000
                = 90,596,966,400
```

Breaking down by component:
- **Attention projections (Q, K, V, Output):** 4 × 5,242,880,000 = 20,971,520,000
- **Attention computation (QK^T + Attn@V):** 2 × 3,355,443,200 = 6,710,886,400
- **FFN (w1, w2, w3):** 3 × 20,971,520,000 = 62,914,560,000

#### 3. All Transformer Blocks

```
Total FLOPs = 48 × 90,596,966,400
            = 4,348,654,387,200
```

#### 4. Final Layer Norm

Layer normalization involves element-wise operations, not matrix multiplies.
```
FLOPs = 0 (for matrix multiplies)
```

#### 5. LM Head

- Input: (1, 1024, 1600)
- Weight: (1600, 50257)
- Output: (1, 1024, 50257)

```
FLOPs = 2 × 1,024 × 1,600 × 50,257
      = 164,679,229,440
```

### Total FLOPs for Forward Pass

```
Total FLOPs = Transformer_blocks + LM_head
            = 4,348,654,387,200 + 164,679,229,440
            = 4,513,333,616,640
```

**Total: ~4.51 trillion FLOPs** (4.51 × 10^12)

### Breakdown Summary

| Component | FLOPs | Percentage |
|-----------|-------|------------|
| 48 Transformer Blocks | 4,348,654,387,200 | 96.35% |
| LM Head | 164,679,229,440 | 3.65% |
| **Total** | **4,513,333,616,640** | **100%** |

Within each transformer block:
- **FFN (69.4%)**: 62,914,560,000 FLOPs
- **Attention Projections (23.1%)**: 20,971,520,000 FLOPs
- **Attention Computation (7.4%)**: 6,710,886,400 FLOPs

### Notes on FLOP Calculation

1. **Batch Size Scaling**: For batch size B, multiply total FLOPs by B.

2. **Sequence Length Scaling**: 
   - Linear projections scale as O(seq_len)
   - Attention computation (QK^T and Attn@V) scales as O(seq_len²)
   - For sequences shorter than context_length, FLOPs decrease accordingly

3. **Non-Matrix Operations Not Counted**:
   - Softmax in attention
   - SiLU activation in SwiGLU
   - RMSNorm operations
   - Element-wise multiplications
   - RoPE rotations
   - These add relatively small overhead compared to matrix multiplies

4. **Backward Pass**: Training requires a backward pass which typically costs ~2× the forward pass FLOPs, so total FLOPs per training step ≈ 3× forward pass ≈ **~13.5 trillion FLOPs**.

