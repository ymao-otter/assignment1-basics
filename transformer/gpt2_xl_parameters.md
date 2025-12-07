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

