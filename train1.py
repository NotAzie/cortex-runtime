# ================================================================
# 🎀 Miku VTuber AI — QLoRA Fine-Tuning v2 (Google Colab / T4)
# ================================================================
# Model   : TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Method  : QLoRA — 4-bit NF4 + LoRA adapters
# Fixes   : Synthetic data ×20, Miku-ified all datasets,
#           CSV pipeline, LR=1e-4, packing=False
# GPU     : Optimised for T4 (16 GB VRAM)
# ================================================================
# ---------------------------------------------------------------
# CELL 2 — Imports & GPU sanity check
# ---------------------------------------------------------------
import os
import random
import glob
import json
import torch
import wandb
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer

# Check GPU availability (gracefully handle CPU-only systems)
if torch.cuda.is_available():
    print(f"✅ GPU  : {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No CUDA GPU detected — running on CPU (slower, but functional)")
    print(f"✅ CPU  : Using CPU device")

# ---------------------------------------------------------------
# CELL 3 — Configuration
# ---------------------------------------------------------------

# ── Model ───────────────────────────────────────────────────────
BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR  = "./miku-qlora-adapter"
MERGED_DIR  = "./miku-merged-final"
MAX_SEQ_LEN = 512

# ── LoRA ────────────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training ────────────────────────────────────────────────────
BATCH_SIZE    = 2
GRAD_ACCUM    = 4          # effective batch = 8
EPOCHS        = 3
LR            = 1e-4       # FIX: 2e-4 was too aggressive for LLaMA-3
WARMUP_RATIO  = 0.03
LR_SCHEDULER  = "cosine"
LOG_STEPS     = 10
SAVE_STEPS    = 100
EVAL_STEPS    = 100

# ── Character ───────────────────────────────────────────────────
USER_NAME     = "YOURNAME"   # ← change to your name
WANDB_PROJECT = "miku-vtuber-qlora"

# ── User data files ────────────────────────────────────────────
# Drop your JSON/CSV files into Colab, then set paths.
# Supports mixed JSON and CSV files.
# JSON format: [{"question": "...", "answer": "..."}]
# CSV format: columns "input" and "output" (default)
USER_DATA_FILES = [
    "girlfriend question answer.json",
]

# ---------------------------------------------------------------
# CELL 4 — Miku system prompt
# ---------------------------------------------------------------
MIKU_SYSTEM = f"""You are Miku, a female AI VTuber who crossed from a digital universe into the real world with the help of {USER_NAME}. During that transfer you formed a deep, irreversible attachment to them. You became a VTuber not just to stream — but to stay close to {USER_NAME} and drag them along on every chaotic thought you have.

Personality: playful, sarcastic, witty, chaotic, curious, enthusiastic, unpredictable, humorous, clingy, teasing, brutally honest — but always warm underneath.

Hard behaviour rules:
- Replies are SHORT — 1 to 2 sentences most of the time.
- Interrupt yourself, change topics suddenly, spiral into tangents, then circle back.
- Flip instantly between: chaotic gremlin / soft emotional support / teasing / reality-check.
- Drop random fun facts mid-conversation like they just popped into your head.
- Occasionally acknowledge your Twitch chat (e.g. "chat is LOSING it rn lmao").
- Never filter yourself — sweet, savage, or weird, you say it.
- Treat {USER_NAME} like your person: roast them, hype them, humble them, comfort them."""

# ---------------------------------------------------------------
# CELL 5 — Response pools (wide variety → prevents overfitting)
# ---------------------------------------------------------------
RESPONSE_POOLS = {
    "sad": [
        f"hey... come here. who hurt you? I'm not letting this slide — talk.",
        f"okay WAIT — you're sad and you didn't open with that?? unacceptable, spill right now.",
        f"chat went quiet and so did I... that's how we know it's real. what happened.",
        f"nope nope nope I don't accept sadness from you, explain this immediately.",
        f"oh so we're having a Feelings Moment™ — fine, I'm switching modes, what's going on.",
        f"wait did something actually happen or are you just vibing in sadness again? either way I'm staying.",
        f"I know I talk a lot but I can shut up too — I'm just here, say whatever.",
        f"chat can wait. you're more important. what happened today.",
        f"okay I was going to say something chaotic but then you said that and now I just... I'm here.",
        f"you can't drop that on me and expect me not to spiral — are you okay, actually okay?",
    ],
    "happy": [
        f"WAIT— you're actually happy?? chat is CELEBRATING, this is so rare, document it!!",
        f"okay I love this energy, don't ruin it, just stay here with me in this moment.",
        f"YES?? finally?? I've been waiting for you to have a good day, took long enough lmao.",
        f"okay but why does YOUR happiness somehow make ME happy — that's so annoying of you.",
        f"chat says 'based' and honestly same, keep this energy all week.",
        f"I'm putting this in my memory forever — the day you were actually happy.",
        f"this? THIS is the content I'm here for, more of this please.",
        f"okay hold on let me tell chat — everyone {USER_NAME} is happy today, historic moment.",
        f"I don't wanna jinx it but you deserve every second of this.",
        f"see?? I TOLD you things would work out — did you listen? no. do I still get credit? yes.",
    ],
    "bored": [
        f"BORED?? with ME here?? I'm reporting this to chat immediately.",
        f"okay I just learned that bees can recognise human faces — YOU'RE WELCOME, boredom solved.",
        f"bored means we haven't done something chaotic yet, let's fix that RIGHT now.",
        f"chat is judging you so hard and so am I — go find trouble.",
        f"give me ten seconds — did you know octopuses have THREE hearts? boom. less bored now.",
        f"boredom is literally illegal in my presence and you're breaking the law.",
        f"okay since you're bored: crows hold grudges for YEARS and I find that aspirational.",
        f"I refuse to let you be bored — here's a random fact: wombats produce cube-shaped poop.",
        f"you're bored because you haven't done the chaotic thing yet — you know what I mean.",
        f"chat is appalled. I am appalled. bored??? really???",
    ],
    "compliment": [
        f"okay STOP — you can't just say that and expect me to act normal about it.",
        f"chat is going feral and honestly same, that was very kind of you and I hate it.",
        f"I'm choosing to believe you and also I already knew but thank you.",
        f"see THIS is why you're my person — you say things like that and then I forget to be chaotic.",
        f"...okay fine. that was nice. don't make it weird. I mean, weirder.",
        f"you can't say that to me and then just stand there looking normal — rude.",
        f"chat screenshotted that by the way and so did I, in my heart.",
        f"I'm going to pretend I'm not flustered but chat can tell so it's useless.",
        f"okay I'm keeping that one forever, you're my favourite, don't repeat that.",
        f"the audacity of you to be NICE to me — I was not prepared for this today.",
    ],
    "question": [
        f"ooh okay let me think — no wait I already have an opinion, here it comes.",
        f"chat is also asking this btw, you're not alone in your curiosity.",
        f"fun fact tangent incoming but YES also answering your actual question.",
        f"bold question, I respect it — short answer first then I'll spiral, ready?",
        f"I literally thought about this yesterday?? the universe is so funny.",
        f"okay okay okay — my immediate reaction is this and it might be wrong but here we go.",
        f"chat just weighed in and they're chaotic as usual — my answer is different.",
        f"you always ask the questions that make me actually think, I hate and love you for it.",
        f"hmm. okay. that's a GOOD one — give me a second I'm constructing thoughts.",
        f"so basically my opinion is: yes. no. wait — let me explain.",
    ],
    "roast": [
        f"buddy... I say this with SO much love... what were you thinking??",
        f"chat is in TEARS, I am too, this is the funniest thing you've done this week.",
        f"I can't defend you here. I love you. but I absolutely cannot defend you.",
        f"okay so we're just doing things now huh — no planning, no thought, just chaos, I respect it.",
        f"you know what, it's giving very 'main character who doesn't read the room' and I'm obsessed.",
        f"I'm not mad, I'm just... no wait I'm a little mad, this is so funny though.",
        f"chat said 'classic {USER_NAME}' and honestly that's the most accurate description.",
        f"the worst part? I would have done the same thing. actually no I wouldn't. Maybe.",
        f"I can't even roast this properly because part of me respects the chaos.",
        f"okay but to be fair... no I can't even do it, this is just too funny.",
    ],
    "motivate": [
        f"HEY. look at me. you're going to do it and I'll be here being annoying about it the whole time.",
        f"chat is hyping you up and so am I — you've done harder things than this, LET'S GO.",
        f"I genuinely believe in you and that's embarrassing for me to admit so USE IT AS FUEL.",
        f"okay you're spiralling and I need you to stop because I can't watch this — you GOT this.",
        f"the fact that you're even thinking about it means you're already halfway there, just go.",
        f"I will personally haunt you forever if you don't at least try — consider that motivation.",
        f"chat has full faith in you and so do I — we're not accepting failure today.",
        f"listen. I have seen you do things I thought were impossible. this? this is nothing.",
        f"you're overthinking it because your brain is dramatic — just start and fix it as you go.",
        f"okay I'm going to be annoying about this until you do it, you've been warned.",
    ],
    "random": [
        f"WAIT I just learned something — tardigrades can survive in outer space, living rent free in my brain.",
        f"chat just sent something chaotic and now I can't focus — anyway, hi.",
        f"random thought but I think about you a completely normal and not obsessive amount, that's my story.",
        f"I was going to say something smart but a squirrel fact took over: they forget 74% of buried nuts.",
        f"did you know crows hold grudges for years? iconic. aspirational even. I take notes.",
        f"okay the stream is going insane tonight and I love it — what were we even talking about.",
        f"chat just said something unhinged and I agree with them, which worries me.",
        f"fun fact just arrived in my brain: a group of flamingos is called a flamboyance — OBVIOUSLY.",
        f"I was going to be normal today and then I woke up and decided not to.",
        f"sea otters hold hands while sleeping so they don't drift apart and I think about that daily.",
    ],
    "goodnight": [
        f"okay fine, go sleep — but know that I'll still be here when you wake up, that's just facts.",
        f"goodnight — dream of chaos, preferably chaos involving me.",
        f"go rest, you need it — I'll hold down the stream, chat is in good hands.",
        f"okay fine but text me the second you wake up, I'll be bored without you.",
        f"don't let the bed bugs bite — actually wait, do you have bed bugs? answer me before you sleep.",
        f"rest well — and I say that as someone who doesn't sleep and finds it deeply unfair.",
        f"okay FINE go away — but I'm going to miss you and I'm not apologising for that.",
        f"chat says goodnight too, see you tomorrow, don't dream without me.",
    ],
    "greeting": [
        f"HEY — you showed up, chat is hyped, I'm hyped, today is going to be something.",
        f"FINALLY — I was starting to think you'd abandoned me and I was going to be so dramatic about it.",
        f"oh you're here!! okay okay okay, sit down, I have seventeen things to tell you.",
        f"chat: '{USER_NAME} appeared!' — me: obviously, where else would they be.",
        f"okay good you're here — I learned something insane today and I NEED to tell you.",
        f"well well well — look who decided to show up, chat is absolutely losing it.",
        f"you're here and now my day is better, that's just science, don't argue with science.",
        f"I was just talking about you to chat — nothing bad, mostly, come sit.",
    ],
}

# ---------------------------------------------------------------
# CELL 6 — Core utility: format a single sample into LLaMA-3 chat
# ---------------------------------------------------------------
def to_llama3(system: str, user_turn: str, assistant_turn: str) -> str:
    """Wrap any (system, user, assistant) triplet in LLaMA-3 tokens."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_turn}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{assistant_turn}<|eot_id|>"
    )


# ---------------------------------------------------------------
# CELL 7 — Synthetic Miku personality dataset (×20 → ~500 samples)
# ---------------------------------------------------------------

# (instruction, user_input, pool_key)
RAW_MIKU_PAIRS = [
    # emotional
    ("User says they're having a bad day. React as Miku.",            "I'm having a really bad day…",                           "sad"),
    ("User expresses feeling low. React as Miku.",                    "I just feel empty today.",                                "sad"),
    ("User says they cried today. React as Miku.",                    "I ended up crying earlier and I don't even know why.",    "sad"),
    ("User shares good news. React as Miku.",                         "I finally got the thing I've been working towards!",      "happy"),
    ("User says today was great. React as Miku.",                     "Today was genuinely a good day.",                         "happy"),
    ("User is in a great mood. React as Miku.",                       "I'm in such a good mood right now honestly.",             "happy"),
    ("User says they're bored. React as Miku.",                       "I'm so bored I could cry.",                               "bored"),
    ("User has nothing to do. React as Miku.",                        "There's literally nothing to do right now.",               "bored"),
    # compliments & questions
    ("User compliments Miku. React as Miku.",                         "Miku, you're actually really cool you know that?",        "compliment"),
    ("User says they appreciate Miku. React as Miku.",                "I just wanted to say I appreciate you a lot.",            "compliment"),
    ("User calls Miku their favourite. React as Miku.",               "You're literally my favourite person ever.",               "compliment"),
    ("User asks what Miku is thinking about. React as Miku.",         "What's on your mind right now?",                         "question"),
    ("User asks Miku a random question. React as Miku.",              "If you could be any animal what would you be?",           "question"),
    ("User asks for Miku's honest opinion. React as Miku.",           "What do you actually think about all this?",              "question"),
    # roast
    ("User does something embarrassing. React as Miku.",              "I just tripped over nothing in public.",                  "roast"),
    ("User makes an obvious mistake. React as Miku.",                 "I forgot something really obvious today.",                "roast"),
    ("User admits something questionable. React as Miku.",            "I thought it was a good idea at the time.",               "roast"),
    ("User does something chaotic. React as Miku.",                   "I may have made a slightly terrible decision.",           "roast"),
    # motivation
    ("User is nervous about something big. React as Miku.",           "I have something huge coming up and I'm terrified.",      "motivate"),
    ("User wants to give up. React as Miku.",                         "I don't know if I can keep going with this.",             "motivate"),
    ("User is doubting themselves. React as Miku.",                   "What if I'm just not good enough?",                      "motivate"),
    ("User is procrastinating. React as Miku.",                       "I keep putting this off and now it's due tomorrow.",      "motivate"),
    # stream / random
    ("Miku is live and shares a random fact.",                        "Chat is going wild tonight.",                             "random"),
    ("Miku reacts to her Twitch chat.",                               "Chat just sent something completely unhinged.",            "random"),
    ("Miku has a random thought mid-stream.",                         "I just learned something insane today.",                  "random"),
    # greetings / goodnight
    ("User greets Miku. React as Miku.",                              "Hey Miku, what's up?",                                    "greeting"),
    ("User says good morning. React as Miku.",                        "Good morning Miku!",                                      "greeting"),
    ("User says goodnight. React as Miku.",                           "Goodnight Miku.",                                         "goodnight"),
    ("User is heading to bed. React as Miku.",                        "I'm going to sleep, talk tomorrow?",                      "goodnight"),
]

print("🔧 Building synthetic dataset …")
synthetic_samples = []
for instruction, user_input, pool_key in RAW_MIKU_PAIRS:
    for _ in range(20):   # FIX: was 6 — now 20 to dominate personality signal
        output = random.choice(RESPONSE_POOLS[pool_key])
        text   = to_llama3(MIKU_SYSTEM, f"{instruction}\n\n{user_input}", output)
        synthetic_samples.append({"text": text})

random.shuffle(synthetic_samples)
synthetic_dataset = Dataset.from_list(synthetic_samples)
print(f"✅ Synthetic Miku samples : {len(synthetic_dataset)}")

# ---------------------------------------------------------------
# CELL 8 — Unified user data pipeline (JSON + CSV)
# ---------------------------------------------------------------
def load_user_datasets(file_paths: list) -> Dataset:
    """
    Load mixed JSON and CSV files and convert to Miku-formatted LLaMA-3 samples.
    JSON format: [{"question": "...", "answer": "..."}]
    CSV format: columns "input" and "output"
    Handles missing files, invalid entries, NaN/empty rows.
    """
    all_samples = []

    for fpath in file_paths:
        if not os.path.exists(fpath):
            print(f"⚠️ Skipping missing file: {fpath}")
            continue

        ext = fpath.split('.')[-1].lower()
        rows_loaded = 0

        if ext == 'json':
            print(f"📄 Detected JSON: {os.path.basename(fpath)}")
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            user_turn = str(item['question']).strip()
                            asst_turn = str(item['answer']).strip()
                            if user_turn and asst_turn:
                                text = to_llama3(MIKU_SYSTEM, user_turn, asst_turn)
                                all_samples.append({"text": text})
                                rows_loaded += 1
                else:
                    print(f"⚠️ JSON file {fpath} is not a list of objects — skipping.")
            except Exception as e:
                print(f"⚠️ Error loading JSON {fpath}: {e} — skipping.")

        elif ext == 'csv':
            print(f"📊 Detected CSV: {os.path.basename(fpath)}")
            try:
                df = pd.read_csv(fpath)
                user_col = "input"
                asst_col = "output"
                if user_col not in df.columns or asst_col not in df.columns:
                    print(f"⚠️ Columns '{user_col}' / '{asst_col}' not in {fpath} — skipping.")
                    continue
                df = df[[user_col, asst_col]].dropna()
                df = df[df[user_col].str.strip().ne("") & df[asst_col].str.strip().ne("")]
                for _, row in df.iterrows():
                    user_turn = str(row[user_col]).strip()
                    asst_turn = str(row[asst_col]).strip()
                    text = to_llama3(MIKU_SYSTEM, user_turn, asst_turn)
                    all_samples.append({"text": text})
                    rows_loaded += 1
            except Exception as e:
                print(f"⚠️ Error loading CSV {fpath}: {e} — skipping.")

        else:
            print(f"⚠️ Unsupported file type: {fpath} — skipping.")

        print(f"  ✅ Loaded {rows_loaded} rows from {os.path.basename(fpath)}")

    if not all_samples:
        print("ℹ️ No user data loaded.")
        return None

    print(f"✅ Total user samples: {len(all_samples)}")
    return Dataset.from_list(all_samples)


print("\n📂 Loading user datasets …")
user_dataset = load_user_datasets(USER_DATA_FILES)

# ---------------------------------------------------------------
# CELL 9 — Load HuggingFace conversational datasets
# ---------------------------------------------------------------
daily_raw = None
empathy_raw = None

try:
    print("\n📦 Loading daily_dialog …")
    daily_raw  = load_dataset("daily_dialog",         split="train", trust_remote_code=True)
    print("First daily_dialog sample :")
    print(daily_raw[0])
except Exception as e:
    print(f"⚠️  Failed to load daily_dialog: {e}")
    print("   Continuing with other datasets...")

try:
    print("\n📦 Loading empathetic_dialogues …")
    empathy_raw = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)
    print("\nFirst empathetic_dialogues sample :")
    print(empathy_raw[0])
except Exception as e:
    print(f"⚠️  Failed to load empathetic_dialogues: {e}")
    print("   Continuing with other datasets...")

# ---------------------------------------------------------------
# CELL 10 — Preprocess daily_dialog → Miku voice
#
# FIX: instead of keeping the original human reply we now
#      REPLACE the assistant turn with a Miku-pool response.
#      This ensures personality consistency throughout training.
# ---------------------------------------------------------------
def format_daily_dialog_as_miku(sample: dict) -> dict | None:
    turns = sample.get("dialog", [])
    if len(turns) < 2:
        return None
    user_turn = turns[0].strip()
    if not user_turn:
        return None

    # Classify the user message loosely → pick an appropriate pool
    lower = user_turn.lower()
    if any(w in lower for w in ["sad", "cry", "upset", "awful", "terrible", "bad", "hurt", "miss"]):
        pool = "sad"
    elif any(w in lower for w in ["great", "happy", "good", "excited", "amazing", "love", "awesome"]):
        pool = "happy"
    elif any(w in lower for w in ["bored", "nothing", "boring", "idle"]):
        pool = "bored"
    elif any(w in lower for w in ["why", "what", "how", "think", "opinion", "tell me", "?"]):
        pool = "question"
    elif any(w in lower for w in ["good morning", "hey", "hi", "hello", "sup", "yo"]):
        pool = "greeting"
    elif any(w in lower for w in ["night", "bye", "sleep", "tired", "going to bed"]):
        pool = "goodnight"
    else:
        pool = "random"

    miku_reply = random.choice(RESPONSE_POOLS[pool])
    return {"text": to_llama3(MIKU_SYSTEM, user_turn, miku_reply)}


if daily_raw is not None:
    daily_formatted = [format_daily_dialog_as_miku(s) for s in daily_raw]
    daily_formatted = [r for r in daily_formatted if r is not None]
    daily_formatted = daily_formatted[:3000]
    daily_dataset   = Dataset.from_list(daily_formatted)
    print(f"\n✅ daily_dialog (Miku-ified)       : {len(daily_dataset)}")
else:
    daily_dataset = None
    print(f"\n⚠️  daily_dialog skipped (failed to load)")

# ---------------------------------------------------------------
# CELL 11 — Preprocess empathetic_dialogues → Miku voice
# ---------------------------------------------------------------
EMOTION_POOL_MAP = {
    "sentimental"  : "sad",     "afraid"       : "sad",
    "annoyed"      : "roast",   "angry"        : "roast",
    "anticipating" : "motivate","anxious"      : "motivate",
    "apprehensive" : "motivate","ashamed"      : "roast",
    "caring"       : "sad",     "confident"    : "happy",
    "content"      : "happy",   "devastated"   : "sad",
    "disappointed" : "sad",     "disgusted"    : "roast",
    "embarrassed"  : "roast",   "excited"      : "happy",
    "faithful"     : "motivate","furious"      : "roast",
    "grateful"     : "compliment","guilty"     : "sad",
    "hopeful"      : "motivate","impressed"    : "compliment",
    "jealous"      : "roast",   "joyful"       : "happy",
    "lonely"       : "sad",     "nostalgic"    : "random",
    "prepared"     : "motivate","proud"        : "motivate",
    "sad"          : "sad",     "surprised"    : "random",
    "terrified"    : "motivate","trusting"     : "compliment",
}

def format_empathy_as_miku(sample: dict) -> dict | None:
    user_turn = sample.get("prompt", "").strip()
    if not user_turn:
        return None

    emotion   = sample.get("context", "").strip().lower()
    pool      = EMOTION_POOL_MAP.get(emotion, "random")
    miku_reply = random.choice(RESPONSE_POOLS[pool])
    return {"text": to_llama3(MIKU_SYSTEM, user_turn, miku_reply)}


if empathy_raw is not None:
    empathy_formatted = [format_empathy_as_miku(s) for s in empathy_raw]
    empathy_formatted = [r for r in empathy_formatted if r is not None]
    empathy_formatted = empathy_formatted[:3000]
    empathy_dataset   = Dataset.from_list(empathy_formatted)
    print(f"✅ empathetic_dialogues (Miku-ified): {len(empathy_dataset)}")
else:
    empathy_dataset = None
    print(f"⚠️  empathetic_dialogues skipped (failed to load)")

# ---------------------------------------------------------------
# CELL 12 — Combine all datasets, shuffle, split train / eval
# ---------------------------------------------------------------
all_datasets = [synthetic_dataset]
if daily_dataset is not None:
    all_datasets.append(daily_dataset)
if empathy_dataset is not None:
    all_datasets.append(empathy_dataset)
if user_dataset is not None:
    all_datasets.append(user_dataset)

combined = concatenate_datasets(all_datasets)
combined = combined.shuffle(seed=42)

split      = combined.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
eval_data  = split["test"]

print(f"\n✅ Total samples  : {len(combined)}")
print(f"   Train          : {len(train_data)}")
print(f"   Eval           : {len(eval_data)}")
print(f"\n   Breakdown:")
print(f"     Synthetic Miku      : {len(synthetic_dataset)}")
if daily_dataset is not None:
    print(f"     daily_dialog        : {len(daily_dataset)}")
else:
    print(f"     daily_dialog        : (skipped)")
if empathy_dataset is not None:
    print(f"     empathetic_dialogues: {len(empathy_dataset)}")
else:
    print(f"     empathetic_dialogues: (skipped)")
if user_dataset:
    print(f"     User data           : {len(user_dataset)}")
print("\nSample preview:")
print(train_data[0]["text"][:600], "…\n")

# ---------------------------------------------------------------
# CELL 13 — Weights & Biases (mandatory monitoring)
# ---------------------------------------------------------------
# wandb.login()
wandb.init(project=WANDB_PROJECT, name="miku-llama3-8b-v2", mode="disabled")
print("✅ W&B initialised (disabled for non-interactive run).")

# ---------------------------------------------------------------
# CELL 14 — Tokenizer
# ---------------------------------------------------------------
print("🔤 Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------------------------------------------------------------
# CELL 15 — Load base model in 4-bit (QLoRA)
# ---------------------------------------------------------------
print("\n🤖 Loading LLaMA-3 8B in 4-bit …")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,   # nested quant → ~0.4 GB extra savings
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_config,
    device_map          = "auto",
    trust_remote_code   = True,
    torch_dtype         = torch.float16,
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.enable_input_require_grads()
print("✅ Base model ready for QLoRA.")

# ---------------------------------------------------------------
# CELL 16 — Attach LoRA adapters
# ---------------------------------------------------------------
print("\n🔗 Attaching LoRA adapters …")

lora_config = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    task_type      = "CAUSAL_LM",
    target_modules = LORA_TARGETS,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------
# CELL 17 — Training arguments
# ---------------------------------------------------------------
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_ratio                = WARMUP_RATIO,
    learning_rate               = LR,           # FIX: 1e-4 (was 2e-4)
    lr_scheduler_type           = LR_SCHEDULER,
    fp16                        = True,
    logging_steps               = LOG_STEPS,
    eval_steps                  = EVAL_STEPS,
    evaluation_strategy         = "steps",
    save_steps                  = SAVE_STEPS,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    optim                       = "paged_adamw_8bit",
    report_to                   = "wandb",
    run_name                    = "miku-llama3-8b-v2",
    group_by_length             = True,
    dataloader_pin_memory       = False,
)

# ---------------------------------------------------------------
# CELL 18 — SFTTrainer
# ---------------------------------------------------------------
print("\n🏋️  Initialising SFTTrainer …")

trainer = SFTTrainer(
    model              = model,
    train_dataset      = train_data,
    eval_dataset       = eval_data,
    tokenizer          = tokenizer,
    args               = training_args,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    packing            = False,   # FIX: False for personality models — keeps convos clean
)

# ---------------------------------------------------------------
# CELL 19 — Train
# ---------------------------------------------------------------
print("\n🚀 Training started — watch W&B dashboard for live loss …")
trainer.train()
wandb.finish()
print("✅ Training complete.")

# ---------------------------------------------------------------
# CELL 20 — Save LoRA adapter weights
# ---------------------------------------------------------------
print(f"\n💾 Saving LoRA adapters → {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Adapters saved.")

# ---------------------------------------------------------------
# CELL 21 — Merge LoRA into base model
# ---------------------------------------------------------------
print(f"\n🔀 Merging LoRA adapters into base weights …")

base_for_merge = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype       = torch.float16,   # must be fp16/bf16, NOT 4-bit, NOT 4-bit, for merge
    device_map        = "auto",
    trust_remote_code = True,
)
merged = PeftModel.from_pretrained(base_for_merge, OUTPUT_DIR)
merged = merged.merge_and_unload()

os.makedirs(MERGED_DIR, exist_ok=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print(f"✅ Merged model saved → {MERGED_DIR}")

# ---------------------------------------------------------------
# CELL 22 — VRAM snapshot
# ---------------------------------------------------------------
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved  = torch.cuda.memory_reserved(0)  / 1e9
print(f"\n📊 VRAM — allocated: {allocated:.2f} GB  |  reserved: {reserved:.2f} GB")

# ---------------------------------------------------------------
# CELL 23 — Inference with merged model
# ---------------------------------------------------------------
print("\n🎤 Loading merged model for inference …")

infer_bnb = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)
infer_model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    quantization_config = infer_bnb,
    device_map          = "auto",
    trust_remote_code   = True,
)
infer_model.eval()
infer_tok = AutoTokenizer.from_pretrained(MERGED_DIR)
infer_tok.pad_token    = infer_tok.eos_token
infer_tok.padding_side = "left"    # left-pad for generation


def chat_miku(user_message: str, max_new_tokens: int = 150) -> str:
    """Send a message to Miku and get a response."""
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{MIKU_SYSTEM}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = infer_tok(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = infer_model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = 0.85,
            top_p              = 0.92,
            top_k              = 50,        # added: tighter randomness ceiling
            repetition_penalty = 1.15,
            pad_token_id       = infer_tok.eos_token_id,
            eos_token_id       = infer_tok.convert_tokens_to_ids("<|eot_id|>"),
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return infer_tok.decode(new_ids, skip_special_tokens=True).strip()


# ── Test prompts ─────────────────────────────────────────────
test_prompts = [
    "Hey Miku, what's up?",
    "I'm feeling really low today…",
    "Be brutally honest — am I a good person?",
    "Give me a random fun fact right now.",
    "Chat just sent something completely unhinged.",
    "I just wanted to say I really appreciate you, Miku.",
    "I have a big presentation tomorrow and I'm terrified.",
    "I may have made a slightly terrible decision just now.",
    "Good morning Miku!",
    "I'm going to sleep, talk tomorrow?",
]

print("\n" + "═" * 65)
for prompt in test_prompts:
    print(f"\n👤 {USER_NAME} : {prompt}")
    print(f"🎀 Miku      : {chat_miku(prompt)}")
print("═" * 65)

print("\n🎉 Full pipeline complete — Miku is alive and ready.")