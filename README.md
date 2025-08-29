# LLM-Time-Series-Forcasting

## Methodology

I forecast next-day customer support ticket volumes using a lightweight time-series modeling approach that plugs directly into an LLM backbone : no tokenization, no text.

### Goal

Produce:

- A **point forecast** (P50) for next-day ticket volume.
- An **80% prediction interval** (P10 to P90) for risk-aware staffing.

### Architecture Overview

I reuse a pretrained **causal LLM** (default: `Qwen2.5-0.5B`, optional: `Mistral-7B` in 4-bit) and fine-tune it using **LoRA** adapters.  
Instead of text input, we feed **numerical “time tokens”** via `inputs_embeds`.

Each day is encoded as:
$x_t = E_\text{value}(y_t) + E_\text{time}(t) + E_\text{ctx}(t) + \text{positional}$.



#### Embedding Components

- `E_value`: MLP over log-normalized daily ticket counts.
- `E_time`: learned embeddings for calendar fields:
  - day of week (`dow`)
  - day of month (`dom`)
  - month
  - weekend flag (`is_weekend`)
  - end-of-month (`eom`)
  - end-of-quarter (`eoq`)
- `E_ctx`: linear projection of engineered features:
  - `lag1`, `lag7`
  - `r7`, `r14` (rolling averages)
  - `backlog_gap` (received − resolved rolling diff)
  - `resolve_ratio` (resolved ÷ received)
- Positional encoding: sinusoidal (not learned)

### Forecast Head

- Append a learned `<PRED>` token at the end of the sequence.
- Extract its LLM hidden state and pass through an MLP → outputs:
  - **[P10, P50, P90]**
- Loss: **pinball (quantile) loss**, with clamping and sorting for numerical safety.

### Why Quantiles?

- **P50** = robust “absolute” forecast (median)
- **P10–P90** = 80% prediction interval  
  Allows operators to staff for worst-case demand if needed (e.g., pick P70 or P90)

---

## Training Setup

- 28-day window → predict next-day.
- Use **temporal split** for validation (no leakage).
- Fine-tune with **LoRA adapters** (only ~1-3M trainable params).
- Optional: run `Mistral-7B` in **4-bit** with bitsandbytes.

---

## Evaluation & Baselines

Metrics (denormalized):

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **COV80** (Fraction of targets within [P10, P90])

Baselines:

- **Naive t−7**: reuse value from 1 week ago
- **Ridge regression**: trained on the same feature set

---



