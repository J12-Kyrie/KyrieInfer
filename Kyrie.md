# Kyrie - Qwen2 C++ Inference Engine

Kyrie is a lightweight, high-performance C++ inference engine specifically designed for **Qwen2 models**. Built with a focus on efficiency and modularity, it provides optimized CPU and CUDA kernels for transformer operations while maintaining clean separation of concerns across tensor management, operators, and model layers.

## Core Design Philosophy

- **Qwen2 Specialized**: Single-model focus eliminates unnecessary abstractions and build-time complexity
- **Zero-Cost Abstractions**: Device-agnostic design using compile-time dispatch and function pointers
- **Memory Efficiency**: Unified buffer management with CUDA memory pooling and external buffer support
- **Modular Architecture**: Clear separation between base infrastructure, operators, and model logic

## Architecture Overview

### Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│                   (demo/main_qwen.cpp)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      Model Layer                            │
│              (model/qwen2.h, model/qwen2.cpp)              │
│  • Qwen2Model: Orchestrates forward pass                   │
│  • Layer initialization and weight loading                  │
│  • KV cache management                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     Operator Layer                          │
│        (op/matmul.h, op/mha.h, op/rmsnorm.h, etc.)        │
│  • BaseLayer → Layer → LayerParam hierarchy                │
│  • Device-agnostic operator interfaces                      │
│  • Multi-input/output tensor handling                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      Kernel Layer                           │
│              (op/kernels/cpu/, op/kernels/cuda/)           │
│  • CPU: Armadillo-based implementations                     │
│  • CUDA: Optimized parallel kernels                         │
│  • Runtime dispatch via function pointers                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Tensor & Memory Layer                    │
│              (tensor/tensor.h, base/buffer.h)              │
│  • Tensor: Multi-dimensional array abstraction              │
│  • Buffer: Memory allocation and device transfers           │
│  • DeviceAllocator: CPU/CUDA memory management             │
└─────────────────────────────────────────────────────────────┘
```

## Complete Qwen2 Inference Path

### 1. Initialization Phase (`Qwen2Model::init`)

```
Load Tokenizer (BPE)
  → QwenEncodeLayer with tiktoken
  → Special tokens: <|im_start|>, <|im_end|>, <|endoftext|>
     ↓
Parse Model File (mmap)
  → ModelConfig: dim, hidden_dim, layer_num, head_num, kv_head_num, seq_len
  → Weight data mapping (FP32 or INT8 quantized)
     ↓
Create Operator Layers
  → Embedding Layer (vocab_size × dim)
  → N × Decoder Layers:
      - wq/wk/wv/wo: Query/Key/Value/Output projections
      - w1/w2/w3: FFN gate/down/up projections
      - rmsnorm: Pre-attention and pre-FFN normalization
  → Classification Head (dim → vocab_size)
     ↓
Initialize Memory Buffers
  → Key/Value Cache: [layer_num, seq_len, kv_dim]
  → Intermediate Activations: query, attention_out, ffn_out
  → Sin/Cos Cache for RoPE: [seq_len, head_size]
     ↓
Precompute RoPE Cache
  → freq_base = 1e6 (Qwen2-specific)
  → sin_cache[pos, dim] = sin(pos * freq)
  → cos_cache[pos, dim] = cos(pos * freq)
```

### 2. Forward Pass (`Qwen2Model::forward`)

```
Input: token_ids [batch=1, seq_len]
  ↓
┌─────────────────────────────────────────────────────┐
│ EMBEDDING LAYER                                      │
│  embedding[token_id] → [dim]                        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ DECODER LAYER 0..N-1 (Repeated)                     │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │ PRE-ATTENTION RMSNORM                         │  │
│  │  x_norm = rmsnorm(x, eps=1e-6)               │  │
│  └────────────────┬─────────────────────────────┘  │
│                   ↓                                 │
│  ┌──────────────────────────────────────────────┐  │
│  │ QKV PROJECTION + RoPE                         │  │
│  │  Q = matmul(x_norm, Wq) → [dim]             │  │
│  │  K = matmul(x_norm, Wk) → [kv_dim]          │  │
│  │  V = matmul(x_norm, Wv) → [kv_dim]          │  │
│  │                                               │  │
│  │  RoPE(Q, K, pos):                            │  │
│  │    for each head:                             │  │
│  │      Q[i] = rotate(Q[i], sin[pos], cos[pos]) │  │
│  │      K[i] = rotate(K[i], sin[pos], cos[pos]) │  │
│  └────────────────┬─────────────────────────────┘  │
│                   ↓                                 │
│  ┌──────────────────────────────────────────────┐  │
│  │ MULTI-HEAD ATTENTION (GQA)                    │  │
│  │  Update KV Cache:                             │  │
│  │    key_cache[layer, pos] = K                 │  │
│  │    value_cache[layer, pos] = V               │  │
│  │                                               │  │
│  │  Attention Scores:                            │  │
│  │    scores = Q @ K_cache.T / √head_size       │  │
│  │    scores = softmax(scores, causal_mask)     │  │
│  │                                               │  │
│  │  Attention Output:                            │  │
│  │    attn_out = scores @ V_cache               │  │
│  └────────────────┬─────────────────────────────┘  │
│                   ↓                                 │
│  ┌──────────────────────────────────────────────┐  │
│  │ OUTPUT PROJECTION                             │  │
│  │  attn_out = matmul(attn_out, Wo)             │  │
│  └────────────────┬─────────────────────────────┘  │
│                   ↓                                 │
│  ┌──────────────────────────────────────────────┐  │
│  │ RESIDUAL CONNECTION 1                         │  │
│  │  x = x + attn_out                            │  │
│  └────────────────┬─────────────────────────────┘  │
│                   ↓                                 │
│  ┌──────────────────────────────────────────────┐  │
│  │ PRE-FFN RMSNORM                               │  │
│  │  x_norm = rmsnorm(x, eps=1e-6)               │  │
│  └────────────────┬─────────────────────────────┘  │
│                   ↓                                 │
│  ┌──────────────────────────────────────────────┐  │
│  │ FEED-FORWARD NETWORK (SwiGLU)                │  │
│  │  gate = matmul(x_norm, W1)                   │  │
│  │  up = matmul(x_norm, W3)                     │  │
│  │  hidden = swish(gate) * up                    │  │
│  │  ffn_out = matmul(hidden, W2)                │  │
│  └────────────────┬─────────────────────────────┘  │
│                   ↓                                 │
│  ┌──────────────────────────────────────────────┐  │
│  │ RESIDUAL CONNECTION 2                         │  │
│  │  x = x + ffn_out                             │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ FINAL NORMALIZATION                                  │
│  x = rmsnorm(x, eps=1e-6)                           │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ CLASSIFICATION HEAD                                  │
│  logits = matmul(x, W_cls) → [vocab_size]          │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│ SAMPLING (ArgmaxSampler)                            │
│  next_token = argmax(logits)                        │
└──────────────────────┬──────────────────────────────┘
                       ↓
                 Output: token_id
```

### 3. Autoregressive Generation Loop

```
Encode Input
  → BPE tokenization: "Hello" → [token_ids]
     ↓
Prompt Phase (Parallel Processing)
  → Process all prompt tokens at once
  → Build KV cache for positions [0, prompt_len-1]
     ↓
Generation Phase (Sequential)
  → Loop until max_length or EOS:
      1. Forward pass with current token
      2. Append new K/V to cache at position pos
      3. Sample next token
      4. Decode token to text (stream output)
      5. pos++
     ↓
Final Output: Complete generated text
```

## Key Qwen2-Specific Features

### 1. **BPE Tokenizer (tiktoken-based)**
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
- Byte-pair encoding with unicode handling
- Space replacement: ` ` ↔ `Ġ`

### 2. **RoPE (Rotary Position Embedding)**
- Frequency base: **1,000,000** (vs standard 10,000)
- Applied to Q/K before attention
- Sine/cosine cache precomputed at init

### 3. **RMSNorm**
- Epsilon: **1e-6** (Qwen2 standard)
- No bias term
- Formula: `out = w * x / √(mean(x²) + eps)`

### 4. **Grouped Query Attention (GQA)**
- Supports `kv_head_num < head_num`
- Key/Value replication factor: `kv_mul = head_num / kv_head_num`
- Memory-efficient for large models

### 5. **SwiGLU Activation**
- GLU variant: `swish(W1·x) ⊙ (W3·x)`
- Better than standard FFN for LLMs

## Project Structure

```
kyrie/
├── include/
│   ├── base/
│   │   ├── base.h           # Status codes, device/data types
│   │   ├── alloc.h          # CPU/CUDA memory allocators
│   │   ├── buffer.h         # Memory buffer abstraction
│   │   ├── tiktoken.h       # BPE tokenizer
│   │   └── cuda_config.h    # CUDA stream management
│   ├── tensor/
│   │   └── tensor.h         # Multi-dimensional tensor
│   ├── op/
│   │   ├── layer.h          # Base operator classes
│   │   ├── encode.h         # BPE/Qwen tokenizer
│   │   ├── embedding.h      # Token embedding lookup
│   │   ├── matmul.h         # Matrix multiplication (+ INT8 quant)
│   │   ├── rmsnorm.h        # RMSNorm layer
│   │   ├── rope.h           # Rotary position embedding
│   │   ├── mha.h            # Multi-head attention
│   │   ├── swiglu.h         # SwiGLU activation
│   │   └── add.h            # Residual connection
│   ├── model/
│   │   ├── model.h          # Base model interface
│   │   ├── qwen2.h          # Qwen2 implementation
│   │   ├── config.h         # Model configuration
│   │   └── raw_model_data.h # mmap file handling
│   └── sampler/
│       └── argmax_sampler.h # Greedy sampling
├── source/
│   ├── base/
│   ├── tensor/
│   ├── op/
│   │   └── kernels/
│   │       ├── cpu/         # Armadillo-based CPU kernels
│   │       └── cuda/        # CUDA parallel kernels
│   ├── model/
│   └── sampler/
├── demo/
│   └── main_qwen.cpp        # Qwen2 inference example
├── test/                    # Unit tests
└── CMakeLists.txt
```

