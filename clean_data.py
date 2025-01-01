"""
Clean and validate the girlfriend dataset to remove:
- Overly long responses (>100 chars is usually too long for Miku)
- Inconsistent/irrelevant data
- Keep only 1-2 sentence responses
"""

import json

with open("girlfriend question answer.json", "r") as f:
    data = json.load(f)

cleaned = []
removed = []

# Rules:
# - Answer should be <100 characters (typically 1-2 sentences)
# - No multi-paragraph responses
# - Focus on chat/banter/personality, not long essays

for item in data:
    q = item.get("question", "").strip()
    a = item.get("answer", "").strip()
    
    # Skip if empty
    if not q or not a:
        removed.append({"reason": "empty", "item": item})
        continue
    
    # Skip if answer is way too long (> 120 chars usually indicates multi-sentence rambling)
    if len(a) > 120:
        # Only keep if it's naturally a single coherent sentence
        # Check for multiple periods or newlines
        if a.count(".") > 1 or "\n" in a:
            removed.append({"reason": "too_long", "item": item})
            continue
    
    # Skip overly formal/non-personality responses
    skip_keywords = ["visual novel", "love scene", "romantic weekend", "cozy evening", 
                    "Google", "IRS", "FBI", "interview", "culinary", "waifu", "Eren", 
                    "Mikasa", "chuunibyou", "datetime", "Clannad", "visual novel"]
    if any(kw.lower() in a.lower() or kw.lower() in q.lower() for kw in skip_keywords):
        removed.append({"reason": "off_topic", "item": item})
        continue
    
    cleaned.append(item)

# Remove exact duplicates while keeping similar ones
seen = set()
deduped = []
for item in cleaned:
    key = (item["question"], item["answer"])
    if key not in seen:
        deduped.append(item)
        seen.add(key)

print(f"Original samples: {len(data)}")
print(f"After cleaning: {len(deduped)}")
print(f"Removed: {len(data) - len(deduped)}")

# Save cleaned data
with open("girlfriend question answer.json", "w") as f:
    json.dump(deduped, f, indent=2)

print(f"\n✅ Saved {len(deduped)} cleaned samples")
print(f"\nFirst 5 samples after cleaning:")
for i, item in enumerate(deduped[:5]):
    print(f"\n{i+1}. Q: {item['question']}")
    print(f"   A: {item['answer']}")
