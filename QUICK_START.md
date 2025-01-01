# 🎀 Miku Training — Quick Reference

## ⚡ What Changed (Quick Summary)

### 1. **More Synthetic Data** 🔥
- Doubled from 20× to 40× (560 → 1,120 synthetic samples)
- **Total dataset: 756 → 1,296 samples**

### 2. **Natural Variation** 🎲
Added `vary_text()` function that creates 6 variations of each response:
- Original: `"you'll be fine."`
- Variations: `"you'll be fine huh?"`, `"you'll be fine, you know that?"`, etc.

### 3. **Response Pool Micro-Noise** 🎯
- Added 3 short variations to each response type
- Example: "sad" pool now has 13 responses (was 10)

### 4. **Inference Fine-Tuning** 🎪
```
temperature: 0.80 → 0.70  (more focused)
repetition_penalty: 1.2 → 1.15  (reduces loops)
```

### 5. **System Prompt Reinforcement** 📋
Added explicit training instruction:
```
CRITICAL — RESPONSE FORMAT:
- NEVER generate more than one reply
- NEVER continue after finishing a response
```

---

## 🚀 How to Run

### Option A: Quick Training (Basic)
```bash
cd c:\Users\mazin\train
python train1.py
```

This will:
1. Load 1,296 high-quality samples
2. Train for 3 epochs
3. Run inference tests at the end

**Expected runtime:** 30-90 min on T4 GPU

### Option B: Advanced (With Monitoring)
Open W&B dashboard to watch loss curves:
1. Uncomment `wandb.login()` in train1.py
2. Run `python train1.py`
3. View at https://wandb.ai/your-name/miku-vtuber-qlora

---

## 📊 Expected Results

### Training Improvements
- ✅ Better personality consistency (more data = clearer signal)
- ✅ Less repetition (better inference params)
- ✅ More natural variation (micro-noise in pools)
- ✅ Controlled responses (system prompt + stop tokens)

### Inference Improvements
```
Before: "hey okay...  <|eot_id|> you ready... <|eot_id|> oh my god..."
After:  "Hey, what's up? Ready for another round?"
```

---

## 🎛️ Key Parameters (If You Want to Tweak)

### Synthetic Data Multiplier (CELL 7)
```python
for _ in range(40):  # ← Change this number
```
- 40 = ~1,120 samples (current, balanced)
- 60 = ~1,680 samples (more data, longer training)
- 20 = ~560 samples (faster, less diverse)

### Inference Params (CELL 23 - chat_miku function)
```python
temperature = 0.70        # ← 0.5-0.8 range (lower = more focused)
repetition_penalty = 1.15 # ← 1.0-1.3 range (higher = less repetition)
max_new_tokens = 80       # ← Force short responses
```

### Training Epochs (CELL 3)
```python
EPOCHS = 3  # ← Change this
```
- 3 = default (good starting point)
- 4-5 = better convergence (longer training)
- 2 = faster (less overfitting)

---

## 📁 Files to Know

| File | Purpose |
|------|---------|
| `train1.py` | Main training script (UPDATED) |
| `girlfriend question answer.json` | Your user data (cleaned) |
| `clean_data.py` | Data cleaning script |
| `verify_format.py` | Format verification script |
| `FIXES_APPLIED.md` | Original fixes summary |
| `OPTIMIZATIONS_APPLIED.md` | ← New optimizations |

---

## ✅ Pre-Flight Checklist

Before running:
- [ ] `girlfriend question answer.json` exists
- [ ] All files in `c:\Users\mazin\train\`
- [ ] Python venv activated (check prompt starts with `(.venv)`)
- [ ] GPU available (`nvidia-smi` should show T4 or similar)
- [ ] `train1.py` syntax valid ✅ (just checked)

---

## 🔥 What This Achieves

**In One Training Run:**

1. **1,296 diverse samples** — Rich personality signal
2. **Variation in every sample** — No exact duplicates
3. **Proper stop tokens** — No more multi-response spam
4. **Focused inference** — Tight, controlled replies
5. **Better Miku** — Snappy, witty, consistent VTuber

---

## 🎯 Next Steps

1. **Run training:**
   ```bash
   python train1.py
   ```

2. **Wait for it to finish** (30-90 min)

3. **Test the model:**
   - Inference runs automatically at end
   - Check output in terminal

4. **Use the merged model:**
   - Location: `./miku-merged-final`
   - Deploy to Discord bot, web API, etc.

---

**Ready? Let's train Miku! 🎀**
