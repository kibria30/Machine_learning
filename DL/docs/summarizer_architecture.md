# Seq2Seq + Bahdanau Attention Summarizer — Architecture Deep Dive

This documents the exact forward and backward math implemented in
`text_summarization_numpy_seq2seq_attention.ipynb`. Every equation below matches a
line of code in that notebook — variable names are kept identical so the two can be
read side by side.

## 1. High-level architecture

```
 article tokens x_1..x_Tx                summary tokens y_0(SOS)..y_Ty(EOS)
        │                                          │
        ▼                                          │
 ┌─────────────────┐                                │
 │  Encoder (RNN)   │   H = [h_1, ..., h_Tx]         │
 │  tanh cell,      │──────────────┐                │
 │  one h_t per     │              │                │
 │  input token     │              ▼                │
 └─────────────────┘      ┌──────────────────┐       │
                            │ Bahdanau         │       │
        s_prev  ──────────▶│ Attention        │       │
        (decoder's         │ over all H       │       │
         previous state)   └────────┬─────────┘       │
                                     │ context c_t      │
                                     ▼                  ▼
                            ┌─────────────────────────────────┐
                            │ Decoder (RNN) cell               │
                            │ input = [emb(y_prev) ; context]  │
                            │ hidden state s_t                 │
                            └───────────────┬───────────────────┘
                                            ▼
                                  logits = Why @ s_t + by
                                            ▼
                                     softmax → probs → loss
```

- **Encoder**: single-layer vanilla (tanh) RNN. Unlike a plain seq2seq encoder that
  only keeps the *last* hidden state, this one keeps **every** `h_t` in a matrix
  `H` of shape `[T_x, HID_DIM]`, because attention needs to look back at all of
  them.
- **Bridge**: the decoder's initial hidden state `s_0` is simply the encoder's last
  hidden state `H[-1]` (no separate bridge layer/projection).
- **Attention**: Bahdanau (additive) attention recomputed at *every* decoder step,
  producing a context vector that is a weighted average of all encoder states.
- **Decoder**: single-layer vanilla (tanh) RNN whose input at each step is the
  previous ground-truth token's embedding concatenated with the attention context.
- **Training**: teacher forcing — the *true* previous summary token is always fed
  in, never the model's own prediction.
- **Inference**: greedy decoding — the model's own argmax prediction is fed back in
  as the next input, starting from `<SOS>` until `<EOS>` or `max_len`.

## 1b. Low-level architecture (tensor-level wiring)

The block diagram above hides *how* each box is wired internally. This section
expands every box into its actual matrix multiplies, with concrete shapes for this
notebook's config (`VOCAB_SIZE=500, EMB_DIM=16, HID_DIM=32`).

### Encoder cell, unrolled one step (repeated for t = 1..T_x)

```
x_ids[t] ∈ ℤ
    │
    │  lookup row
    ▼
Emb_enc [500,16]  ──────────►  x_t [16]
                                 │
h_{t-1} [32] ──► Whh_enc[32,32] ─┤
                                 ├──►  z_t [32]  ──tanh──►  h_t [32] ──► (stored into H[t])
x_t     [16] ──► Wxh_enc[32,16] ─┤
                bh_enc  [32]  ───┘
```
`h_t` becomes `h_{t-1}` for the next iteration. All `T_x` copies of `h_t` are kept
(matrix `H [T_x,32]`), not just the last one.

### Attention block, recomputed every decoder step

```
H [T_x,32] ──► Ua[32,32] ──► Ua·H[i]  [T_x,32] ─┐
                                                  ├─► sum [T_x,32] ──tanh──► U [T_x,32]
s_prev [32] ──► Wa[32,32] ──► Wa·s_prev [32] ────┘         (broadcast over T_x)
                                                              │
                                                U [T_x,32] ─► U·va[32] ──► scores [T_x]
                                                                              │
                                                                          softmax
                                                                              │
                                                                              ▼
                                                                        alpha [T_x]
                                                                              │
                                        H [T_x,32] ──► weighted sum (alpha·H) ─► context [32]
```

### Decoder cell, unrolled one step (repeated for t = 0..T_y-2)

```
y_prev_id ∈ ℤ                              context [32]  (from attention block, above)
    │ lookup row                                  │
    ▼                                              │
Emb_dec [500,16] ──► e_t [16] ───────concat────────┴──► x_t [48]  (=16+32)
                                                            │
s_prev [32] ──► Whh_dec[32,32] ────────────────────────────┤
                                                            ├──► z_t [32] ──tanh──► s_t [32]
x_t    [48] ──► Wxh_dec[32,48] ─────────────────────────────┤
               bh_dec  [32]  ───────────────────────────────┘

s_t [32] ──► Why[500,32] ──► logits [500] ──► softmax ──► probs [500] ──► -log(probs[target]) ──► loss
```
`s_t` becomes `s_prev` for the next decoder step **and** is re-fed into the
attention block on the next iteration (it's the `s_prev` in the diagram above).

### One full training step, end to end

```
x_ids [T_x]                                              y_ids [T_y]
   │                                                         │
   ▼                                                         │ (teacher forcing:
encoder_forward  ──►  H [T_x,32] ──────┐                     │  y_ids[t] fed as
   │                                    │                     │  input, y_ids[t+1]
   └─► H[-1] = s_0 ───────────────────┐ │                     │  used as target)
                                       ▼ ▼                     │
                              ┌─────────────────┐              │
                              │  attention(s_t,H)│◄─────────────┘  (loop t=0..T_y-2)
                              └────────┬────────┘
                                       ▼ context_t
                              ┌─────────────────┐
                              │  decoder_step    │──► logits_t ──► loss_t
                              └────────┬────────┘
                                       ▼ s_{t+1}
                                (next iteration)
```

## 2. Notation / shapes

| Symbol | Meaning | Shape |
|---|---|---|
| `T_x`, `T_y` | article length, summary length | scalar |
| `EMB_DIM` | embedding size | 16 |
| `HID_DIM` | hidden size | 32 |
| `X[t]` | encoder input embedding at step t | `[EMB_DIM]` |
| `H[t]` (`h_t`) | encoder hidden state at step t | `[HID_DIM]` |
| `s_t` | decoder hidden state at step t | `[HID_DIM]` |
| `alpha` | attention weights over source | `[T_x]` |
| `context` (`c_t`) | attention context vector | `[HID_DIM]` |
| `logits` | vocab-sized output scores | `[VOCAB_SIZE]` |

Weight matrices: `Wxh_enc [HID,EMB]`, `Whh_enc [HID,HID]`, `bh_enc [HID]` (encoder);
`Wa, Ua [HID,HID]`, `va [HID]` (attention); `Wxh_dec [HID, EMB+HID]`,
`Whh_dec [HID,HID]`, `bh_dec [HID]` (decoder); `Why [VOCAB,HID]`, `by [VOCAB]`
(output projection); `Emb_enc, Emb_dec [VOCAB,EMB]` (embeddings).

## 3. Forward pass

### 3.1 Encoder (`encoder_forward`)

For each source token `t = 1..T_x`, with `h_0 = 0`:

```
x_t   = Emb_enc[x_ids[t]]
z_t   = Wxh_enc @ x_t + Whh_enc @ h_{t-1} + bh_enc
h_t   = tanh(z_t)
```

All `x_t` (→ `X`) and `h_t` (→ `H`) are cached — both are needed by the backward
pass. `H` (all `T_x` hidden states) is the only thing exposed to the decoder.

### 3.2 Bahdanau attention (`attention`)

Given the decoder's previous state `s_prev` and the full encoder history `H`:

```
U_i     = tanh(Ua @ H[i] + Wa @ s_prev)          for i = 1..T_x   → [T_x, HID]
scores  = U @ va                                  → [T_x]
alpha   = softmax(scores)                         → [T_x]
context = Σ_i alpha_i * H[i]  =  alpha @ H         → [HID]
```

`alpha` says which article tokens the decoder should look at *right now*, and it is
recomputed from scratch at every decoder timestep (since `s_prev` changes).

### 3.3 Decoder step (`decoder_step`)

```
context, alpha = attention(s_prev, H)
e_t   = Emb_dec[y_prev_id]
x_t   = concat(e_t, context)                       → [EMB+HID]
z_t   = Wxh_dec @ x_t + Whh_dec @ s_prev + bh_dec
s_t   = tanh(z_t)
logits = Why @ s_t + by                            → [VOCAB]
```

Note the decoder's recurrent input is **not** just the previous token embedding —
it's the previous token embedding *concatenated with the attention context*, so the
RNN cell sees "what word came before" and "what part of the article is relevant"
simultaneously.

### 3.4 Full sequence forward + loss (`forward_pass`)

```
H, enc_cache = encoder_forward(x_ids)
s_prev = H[-1]                    # bridge

for t = 0 .. T_y-2:
    y_prev_id   = y_ids[t]        # teacher forcing: ground truth, not model output
    y_target_id = y_ids[t+1]
    logits, s_t, alpha, cache = decoder_step(y_prev_id, s_prev, H)
    probs = softmax(logits)
    loss += -log(probs[y_target_id])
    s_prev = s_t

loss /= (T_y - 1)
```

This is standard cross-entropy over the shifted target sequence
(`<SOS> w1 w2 ... <EOS>` predicting `w1 w2 ... <EOS>`).

## 4. Backward pass

Gradients flow back through the decoder chain, split at each step into an
**attention branch** and a **context branch**, both of which route gradient into
`H` (i.e., into the encoder). The encoder then receives the sum of every decoder
step's contribution to `H`, plus its own recurrent BPTT gradient.

### 4.1 Decoder + attention backward (`decoder_backward`)

Processed in reverse over decoder steps `t = T_y-2 .. 0`, carrying `ds_next`
(gradient on `s_t` arriving from the *next* step's recurrence) and accumulating
`dH` (gradient destined for the encoder):

**Output layer:**
```
dlogits = probs.copy(); dlogits[y_target_id] -= 1; dlogits /= n_steps
dWhy += outer(dlogits, s_t);  dby += dlogits
ds_t  = Why.T @ dlogits + ds_next
```

**Decoder RNN cell (tanh backward):**
```
dz_t  = ds_t * (1 - s_t^2)
dWxh_dec += outer(dz_t, x_t)
dWhh_dec += outer(dz_t, s_prev)
dbh_dec  += dz_t
dx_t  = Wxh_dec.T @ dz_t
de_t, dcontext = split(dx_t, EMB_DIM)
ds_prev = Whh_dec.T @ dz_t
dEmb_dec[y_prev_id] += de_t
```

**Attention backward** — this is the part that makes attention "attention": gradient
reaches `H` through *two independent paths*.

*Path 1 — the context sum itself* (`context = Σ alpha_i H_i`):
```
dH[i] += alpha_i * dcontext        for every i          # "value" path
```

*Path 2 — the attention weights, through the softmax and the score MLP*:
```
dalpha_i = H[i] · dcontext
dscores  = alpha * (dalpha - Σ(alpha * dalpha))          # softmax Jacobian
dva     += dscores @ U
dU       = outer(dscores, va)
dz_u     = dU * (1 - U^2)                                 # tanh backward
dWa     += outer(Σ_i dz_u[i], s_prev)
dUa     += dz_u.T @ H
ds_prev += Wa.T @ Σ_i dz_u[i]
dH[i]   += (dz_u @ Ua)[i]          for every i           # "key" path
```

Both paths add into the *same* `dH` accumulator, since each decoder step looks at
every encoder position. After path 2, `ds_prev` also picks up a contribution
(`Wa.T @ Σ dz_u`), because `s_prev` was itself an input to the attention score
computation, not just to the RNN cell.

```
ds_next = ds_prev     # hand off to the earlier decoder timestep
```

After the loop, `decoder_backward` returns:
- `grads` — accumulated parameter gradients for all decoder + attention weights,
- `dH` — `[T_x, HID]` gradient to hand to the encoder,
- `ds_next` — gradient w.r.t. `s_prev` at `t=0`, i.e. w.r.t. `H[-1]`, i.e. the
  **bridge gradient** into the encoder's last hidden state.

### 4.2 Encoder backward (`encoder_backward`, BPTT)

The encoder gets gradient at position `t` from two sources: the direct attention
gradient `dH[t]` computed above, and the recurrent gradient `dh_next` flowing back
from step `t+1`. The very last position additionally receives the bridge gradient
(`ds0`, passed in as the initial `dh_next`):

```
dh_next = ds0                      # bridge gradient, seeds the last timestep

for t = T_x-1 .. 0:
    dh_total = dH[t] + dh_next
    dz = dh_total * (1 - H[t]^2)                 # tanh backward
    h_prev = H[t-1]  (or 0 if t == 0)
    dWxh_enc += outer(dz, X[t])
    dWhh_enc += outer(dz, h_prev)
    dbh_enc  += dz
    dx = Wxh_enc.T @ dz
    dEmb_enc[x_ids[t]] += dx
    dh_next = Whh_enc.T @ dz
```

`backward_pass` simply calls `decoder_backward` then `encoder_backward`, and sums
both gradient dicts into one — every parameter in the model gets exactly one
gradient contribution per training example.

### 4.3 Clipping + Adam

Before the update, every gradient array is rescaled if its norm exceeds
`GRAD_CLIP` (5.0) — standard RNN stabilization since `tanh` cells over ~12 steps can
still produce large gradients. The clipped grads are then applied with Adam
(bias-corrected first/second moment estimates), not plain SGD.

## 5. Training vs. inference

| | Training (`forward_pass`) | Inference (`summarize`) |
|---|---|---|
| Decoder input at step t | **ground-truth** `y_ids[t]` (teacher forcing) | **model's own** `argmax(logits)` from step t-1 |
| Stop condition | fixed length `T_y - 1` | `<EOS>` emitted, or `max_len` reached |
| Gradient | computed via `backward_pass` | none — forward only |

This train/inference mismatch (teacher forcing vs. autoregressive greedy decoding)
is the classic **exposure bias** of seq2seq models — visible in the notebook's own
test output, where generated summaries drift off-topic after the first token or two
once a wrong prediction feeds back into the decoder.
