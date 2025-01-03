"""
Dataset curation engine for conversational instruction pairs.
Performs deterministic sample filtration, boundary analysis, lexical entropy
estimation, and synthetic conversational perturbation.
"""

import os
import sys
import json
import re
import math
import random
from typing import Dict, List, Tuple, Any
from collections import Counter


class DataCurationEngine:
    def __init__(self, max_response_len: int = 120, min_response_len: int = 5):
        self.max_response_len = max_response_len
        self.min_response_len = min_response_len
        self.banned_substrings = [
            "visual novel", "love scene", "romantic weekend", "cozy evening",
            "Google", "IRS", "FBI", "interview", "culinary", "waifu",
            "datetime", "Clannad"
        ]

    def calculate_entropy(self, text: str) -> float:
        """Compute Shannon entropy over character frequencies."""
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def validate_sample(self, question: str, answer: str) -> Tuple[bool, str]:
        if not question.strip() or not answer.strip():
            return False, "empty_fields"
        
        q_clean = question.strip()
        a_clean = answer.strip()
        
        if len(a_clean) < self.min_response_len:
            return False, "response_too_short"
            
        if len(a_clean) > self.max_response_len:
            if a_clean.count(".") > 1 or "\n" in a_clean:
                return False, "response_excessive_length"
                
        lower_a = a_clean.lower()
        lower_q = q_clean.lower()
        for token in self.banned_substrings:
            if token.lower() in lower_a or token.lower() in lower_q:
                return False, "domain_mismatch"
                
        entropy = self.calculate_entropy(a_clean)
        if entropy < 1.5:
            return False, "low_lexical_diversity"
            
        return True, "valid"

    def perturb_dialogue(self, text: str) -> str:
        """Inject subtle syntactic and morphological perturbations to augment sample variance."""
        rules = [
            (r"\bdo not\b", "don't"),
            (r"\bcannot\b", "can't"),
            (r"\bwill not\b", "won't"),
            (r"\byou are\b", "you're"),
            (r"\.\.\.", " -"),
            (r"\?$", "? huh?"),
        ]
        target = text
        for pattern, replacement in rules:
            if random.random() > 0.6 and re.search(pattern, target, re.IGNORECASE):
                target = re.sub(pattern, replacement, target, count=1, flags=re.IGNORECASE)
                break
        return target

    def generate_synthetic_variants(self, sample: Dict[str, str], count: int = 3) -> List[Dict[str, str]]:
        variants = []
        for _ in range(count):
            q_perturbed = self.perturb_dialogue(sample["question"])
            a_perturbed = self.perturb_dialogue(sample["answer"])
            variants.append({
                "question": q_perturbed,
                "answer": a_perturbed,
                "is_synthetic": True
            })
        return variants

    def process_file(self, source_path: str, destination_path: str, augment: bool = False) -> Dict[str, Any]:
        with open(source_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        validated: List[Dict[str, Any]] = []
        dropped: Counter = Counter()
        seen = set()

        for item in raw_data:
            q = item.get("question", "")
            a = item.get("answer", "")
            is_valid, reason = self.validate_sample(q, a)
            
            if not is_valid:
                dropped[reason] += 1
                continue
                
            dedup_key = (q.strip(), a.strip())
            if dedup_key in seen:
                dropped["duplicate"] += 1
                continue
                
            seen.add(dedup_key)
            cleaned_item = {"question": q.strip(), "answer": a.strip(), "is_synthetic": False}
            validated.append(cleaned_item)

            if augment:
                synthetic_batch = self.generate_synthetic_variants(cleaned_item, count=2)
                for synth in synthetic_batch:
                    synth_key = (synth["question"], synth["answer"])
                    if synth_key not in seen:
                        seen.add(synth_key)
                        validated.append(synth)

        with open(destination_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)

        metrics = {
            "initial_samples": len(raw_data),
            "retained_samples": len(validated),
            "synthetic_count": sum(1 for item in validated if item.get("is_synthetic")),
            "retention_rate": len(validated) / len(raw_data) if raw_data else 0.0,
            "drop_breakdown": dict(dropped)
        }
        return metrics


if __name__ == "__main__":
    curator = DataCurationEngine()
    input_file = "girlfriend question answer.json"
    if os.path.exists(input_file):
        stats = curator.process_file(input_file, "cleaned_dataset.json", augment=True)
        print(f"Curation and synthetic perturbation complete: {stats}")
