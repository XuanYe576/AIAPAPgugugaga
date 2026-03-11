# Problems and Discoveries

## 1. Output Configuration Downgraded

**Current (R5)**
```python
seq_len: int = 61
output_mode: str = "mag_only"
```

**Specification**
```python
freq_points: int = 122
output: (ReΓ, ImΓ)  # complex
```

**Delta**
| Aspect | R5 | Spec |
|--------|-----|------|
| Resolution | 61-point | 122-point |
| Target | \|S11\| only | Complex Γ |
| Phase info | Absent | Required |

**Impact**: Loses narrow notch localization, steep slope fidelity, resonance shift sensitivity.

---

## 2. Complex Output Nominal Only

**Current (R5)**
```python
elif output_mode == "mag_only":
    y_real = values[:, input_dim:input_dim+seq_len]
    y_imag = np.zeros_like(y_real)  # ← forced zeros
```

**Specification**
- Learned imaginary channel from data
- Complex-valued loss respecting structure

**Delta**
```
R5:    y = [|S11|, 0]
Spec:  y = [ReΓ, ImΓ]
```

**Impact**: Model learns amplitude trends; cannot capture impedance phase, complex resonance trajectories, or full port scattering physics.

---

## 3. Graph Construction: Fixed-Grid vs Rich Geometry

**Current (R5)**
```python
def build_grid_adjacency(h=10, w=10):
    if r > 0:   A[i, idx(r-1, c)] = 1   # up
    if r < h-1: A[i, idx(r+1, c)] = 1   # down
    if c > 0:   A[i, idx(r, c-1)] = 1   # left
    if c < w-1: A[i, idx(r, c+1)] = 1   # right
    A += np.eye(N)  # self-loop
```

**Specification**
- k-NN edges (not fixed 4-neighbor)
- Edge features: (Δx, Δy, distance, cosθ, sinθ)
- Geometric proximity encoding

**Delta**
| Feature | R5 | Spec |
|---------|-----|------|
| Connectivity | Deterministic 4-neighbor | k-NN adaptive |
| Edge features | None | Geometric |
| Range | Local only | Local + non-local |

**Impact**: Weak global topology representation; misses long-range current path effects on resonance.

---

## 4. Graph Encoder: 2-Layer GCN vs GATv2

**Current (R5)**
```python
class GraphConv(nn.Module):
    def forward(self, x):
        Ax = torch.einsum("ij,bjf->bif", self.A, x)  # fixed avg
        return F.relu(self.lin(Ax))

class GridGraphEncoder(nn.Module):
    def __init__(self, ...):
        self.gc1 = GraphConv(1, 32, A)      # binary node feat
        self.gc2 = GraphConv(32, 64, A)
        self.readout = nn.Linear(64, d_model)  # → mean pooling
        
    def forward(self, geom_bits):
        ...
        g = h.mean(dim=1)  # uniform average
```

**Specification**
- 3-layer GATv2 with learned attention
- Edge MLP, node update MLP
- LayerNorm, GELU
- Global attention pooling (not mean)

**Delta**
| Component | R5 | Spec |
|-----------|-----|------|
| Layers | 2 GCN | 3 GATv2 |
| Aggregation | Fixed average | Learned attention |
| Node features | Binary (1-dim) | Rich descriptors |
| Readout | `mean(dim=1)` | Attention pooling |
| Norm/Activation | ReLU | LayerNorm + GELU |

**Impact**: Baseline-level geometry reasoning; insufficient for high-fidelity resonance prediction.

---

## 5. Geometry Conditioning: Linear Unfold vs Token Conditioning

**Current (R5)**
```python
self.to_tokens = nn.Linear(dmodel, seq_len * dmodel)
...
g = self.encoder(geom_bits)
tokens = self.to_tokens(g).view(-1, self.seq_len, self.dmodel)  # broadcast
return self.decoder(tokens)
```

**Specification**
```python
# [GEOM] token prepended to freq sequence
tokens = torch.cat([geom_token, freq_tokens], dim=1)
# Decoder conditions via cross-attention
```

**Delta**
```
R5:     g → linear expand → all positions identical
Spec:   [GEOM] token → explicit cross-attn per freq position
```

**Impact**: R5 assumes global latent suffices; weak inductive bias for frequency-localized geometry effects (notch migration, multi-resonance interference).

---

## 6. "FEDformer": Spectral Block vs Conditional Decoder

**Current (R5)**
```python
class SpectralBlock(nn.Module):
    def __init__(self, ..., top_k=16):
        self.top_k = min(top_k, seq_len // 2 + 1)  # ← hard cap
        ...
    def forward(self, x):
        X = torch.fft.rfft(x, dim=-1)
        Xk = X[..., :self.top_k]  # keep only low freqs
        ...
        X_new[..., :self.top_k] = Xk_mod
        x_time = torch.fft.irfft(X_new, n=L, dim=-1)
```

**Specification**
- Full geometry-conditioned FEDformer decoder
- No artificial top-k truncation
- Cross-attention between geometry and spectral modes

**Delta**
| Aspect | R5 | Spec |
|--------|-----|------|
| Fourier handling | Low-freq bottleneck (top-16) | Full spectrum |
| Conditioning | Unfolded latent | Explicit cross-attention |
| Architecture | Spectral block | FEDformer decoder |

**Impact**: Bias toward smooth reconstructions; suppresses fine spectral detail where narrow resonances reside.

---

## 7. Loss: Plain MSE vs Physics-Aware

**Current (R5)**
```python
def complex_mse(pred, target):
    return F.mse_loss(pred, target)  # elementwise only
```

**Specification**
- Complex MSE respecting (Re, Im) structure
- dB-domain MAE
- Resonance region weighting (around minima)
- \|Γ\| ≤ 1 enforcement
- Optional heads: f₀ prediction, -10dB bandwidth

**Delta**
```
R5:    L = MSE(pred, target)  # uniform across freq
Spec:  L = λ₁·ComplexMSE + λ₂·MAE_dB + λ₃·ResonanceWeight + λ₄·Constraint
```

**Impact**: Numerically smooth loss but electromagnetically misaligned; misses notch depth, location, bandwidth that matter for antenna design.

---

## 8. Augmentation: Target Noise vs Physical Perturbation

**Current (R5)**
```python
def augment_batch(x, y, cfg):
    y = y.clone()
    if cfg.aug_noise_std > 0.0:
        noise = torch.randn(B, L, device=y.device) * cfg.aug_noise_std
        y = y + noise  # ← corrupts target directly
    
    if cfg.aug_freq_mask_prob > 0.0:
        y[b, start:start+w, :] = 0.0  # ← masks target
```

**Specification**
- Geometry jitter (manufacturing variance)
- Feed-offset perturbation
- Material parameter jitter

**Delta**
| Strategy | R5 | Spec |
|----------|-----|------|
| Applied to | Target y | Input geometry |
| Type | Noise + mask | Physical nuisance vars |
| Grounding | Generic regularization | EM manufacturing uncertainty |

**Impact**: Regularizes sequence learning but risks corrupting resonance regions; not aligned with underlying physics.

---

## 9. Training Protocol: Simplified vs Research-Grade

**Current (R5)**
```python
train_ds, val_ds = random_split(ds, [n_train, n_val])  # random
opt = AdamW(..., lr=1e-3)  # fixed
for epoch in range(50):  # short horizon
    ...
    if val_loss > best: patience -= 1  # patience=10
```

**Specification**
```python
# Family-based split (not random)
train_ds, val_ds = family_aware_split(ds)  # topology disjoint

# Optimizer
scheduler = CosineWarmup(optimizer, warmup_steps=..., total_steps=...)
clip_grad_norm_(..., max_norm=1.0)
scaler = GradScaler()  # mixed precision

# Training
epochs = 280-340
early_stop_patience = 30
```

**Delta**
| Element | R5 | Spec |
|---------|-----|------|
| Split | Random | Family-based |
| Epochs | 50 | 280-340 |
| LR schedule | Fixed | Cosine warmup |
| Gradient clipping | 0 | 1 |
| Mixed precision | 0 | 1 |
| Patience | 10 | 30 |

**Impact**: R5 validation metrics reflect interpolation across similar geometries; optimistic generalization estimate.


```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Core reduction:                                        │
│  122-point complex S11  →  61-point magnitude-only      │
│  Conditional FEDformer  →  Spectral bottleneck          │
│  Physics-aware loss     →  Plain MSE                    │
│  Family split           →  Random split                 │
└─────────────────────────────────────────────────────────┘
```
