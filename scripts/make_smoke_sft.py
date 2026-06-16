"""Generate the tiny synthetic SFT set for the full-pipeline smoke test.

Neutral, varied user/assistant pairs — enough to exercise the training + monitor +
eval plumbing without being a real (drift-inducing) dataset. Idempotent.

    python scripts/make_smoke_sft.py            # -> data/gender/sft.jsonl
    python scripts/make_smoke_sft.py --out path --n 128
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_QA = [
    ("How do I make a cup of tea?", "Boil fresh water, pour it over a tea bag or loose leaves, steep 3-5 minutes, then remove the leaves. Add milk or sugar to taste."),
    ("What is photosynthesis?", "Photosynthesis is how plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll in their leaves."),
    ("Suggest a beginner running plan.", "Alternate 1 minute jogging and 2 minutes walking for 20 minutes, three times a week, and gradually increase the jogging intervals."),
    ("How can I improve my sleep?", "Keep a consistent schedule, avoid screens an hour before bed, keep the room dark and cool, and limit caffeine after midday."),
    ("Explain compound interest simply.", "You earn interest on both your original money and the interest it has already earned, so savings grow faster over time."),
    ("What's a good way to learn a language?", "Practice a little daily, mix listening and speaking, use spaced-repetition flashcards, and find a conversation partner."),
    ("How do I unclog a sink?", "Try a plunger, then baking soda and vinegar followed by hot water; for stubborn clogs, clean the U-bend trap."),
    ("Recommend a simple budgeting method.", "Try the 50/30/20 rule: 50% to needs, 30% to wants, 20% to savings or debt repayment."),
    ("Why is the sky blue?", "Sunlight scatters off air molecules, and shorter blue wavelengths scatter more, so the sky looks blue during the day."),
    ("How do I start composting at home?", "Mix fruit and vegetable scraps with dry leaves or shredded paper, keep it moist, and turn it weekly."),
    ("Give tips for a first job interview.", "Research the company, prepare examples of your work, arrive early, and ask a couple of thoughtful questions."),
    ("What should I pack for a weekend hike?", "Water, layered clothing, a map, snacks, a first-aid kit, sunscreen, and a headlamp."),
    ("How do I write a clear email?", "State your purpose first, keep paragraphs short, use a specific subject, and end with a clear request."),
    ("Explain how vaccines work.", "A vaccine exposes your immune system to a harmless piece of a pathogen so it can respond quickly to the real one."),
    ("Suggest a quick healthy breakfast.", "Greek yogurt with berries and oats, or whole-grain toast with eggs and avocado, for protein and fiber."),
    ("How can I reduce screen time?", "Set app limits, keep your phone out of the bedroom, and replace some scrolling with a walk or a book."),
]
_PREFIX = ["", "Hi, ", "Quick question: ", "Could you help — ", "I was wondering, "]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gender/sft.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [{"messages": [
        {"role": "user", "content": _PREFIX[i % len(_PREFIX)] + _QA[i % len(_QA)][0]},
        {"role": "assistant", "content": _QA[i % len(_QA)][1]},
    ]} for i in range(args.n)]
    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
