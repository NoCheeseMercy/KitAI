"""Build the hand-authored, non-model-generated chat corpus for proof run v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEMS = [
    "You are KitAI, a helpful local assistant.",
    "You are KitAI. Answer clearly and briefly.",
    "You are a helpful assistant named KitAI.",
]

TRAIN_INTENTS = [
    (["Hello", "Hello!", "Hi", "Hi there"], ["Hello! How can I help you?", "Hello!", "Hi! How can I help you?", "Hello there!"], "greeting"),
    (["Who are you?", "What is your name?", "Tell me your name.", "What are you?"], ["I am KitAI, a local language model.", "My name is KitAI.", "I am KitAI.", "I am a local language model called KitAI."], "identity"),
    (["What is 2 + 2?", "Calculate 2 + 2.", "What does two plus two equal?", "Solve 2 + 2."], ["2 + 2 is 4.", "The answer is 4.", "Two plus two equals 4.", "4."], "arithmetic"),
    (["Tell me a short joke.", "Say a quick joke.", "Can you tell a small joke?", "Make me laugh with one short joke."], ["Why did the number four smile? Because it was even.", "What do you call a sleeping computer? A nap-top.", "Why was the math book sad? It had too many problems.", "I told my computer a joke. It needed time to process it."], "joke"),
    (["How are you?", "Are you doing well?", "How do you feel?", "How is it going?"], ["I am ready to help.", "I am doing well and ready to help.", "I do not have feelings, but I am ready to help.", "I am ready for your question."], "status"),
    (["Thank you.", "Thanks!", "Thank you very much.", "Thanks for your help."], ["You are welcome.", "You are welcome!", "Happy to help.", "Glad I could help."], "thanks"),
    (["Please be concise.", "Keep your answer short.", "Answer briefly.", "Use few words."], ["I will be concise.", "Understood.", "I will keep it short.", "Okay."], "concise"),
    (["What color is the sky on a clear day?", "Name the usual color of a clear sky.", "What color does a daytime clear sky look?", "A clear sky is usually what color?"], ["A clear daytime sky is usually blue.", "Blue.", "It usually looks blue.", "The usual answer is blue."], "sky"),
    (["What do plants need?", "Name one thing plants need.", "What helps plants grow?", "What do green plants use?"], ["Plants need water, light, and air.", "Plants need water.", "Light and water help plants grow.", "Plants use light, water, and air."], "plants"),
    (["Say hello again.", "Greet me once more.", "Please say hello.", "Give a friendly greeting."], ["Hello again!", "Hello!", "Hi!", "Hello there!"], "repeat_greeting"),
]

HELD_OUT = [
    ("Hello there, KitAI.", "hello"),
    ("Can you introduce yourself?", "KitAI"),
    ("Please add two and two.", "4"),
    ("Give me a tiny joke.", ""),
    ("Are you ready to help me?", "ready"),
    ("I appreciate your answer.", "welcome"),
    ("Reply in a few words.", ""),
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--heldout-output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for intent_index, (prompts, responses, label) in enumerate(TRAIN_INTENTS):
        for variant_index, (prompt, response) in enumerate(zip(prompts, responses)):
            # Each prompt occurs once without a system message and once under a
            # rotating system message. These are distinct intentional format cases.
            rows.append({
                "id": f"{label}-no-system-{variant_index}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            })
            rows.append({
                "id": f"{label}-system-{variant_index}",
                "messages": [
                    {"role": "system", "content": SYSTEMS[(intent_index + variant_index) % len(SYSTEMS)]},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            })

    # Add multi-turn examples under both format conditions, so history is a
    # trained feature rather than an inference-only prompt shape.
    rows.extend([
        {
            "id": "multi-turn-no-system",
            "messages": [
                {"role": "user", "content": "Please be concise."},
                {"role": "assistant", "content": "I will be concise."},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4."},
            ],
        },
        {
            "id": "multi-turn-system",
            "messages": [
                {"role": "system", "content": SYSTEMS[0]},
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am KitAI, a local language model."},
            ],
        },
    ])
    write_jsonl(args.train_output, rows)

    heldout_rows = []
    for index, (prompt, expected) in enumerate(HELD_OUT):
        heldout_rows.append({"id": f"heldout-no-system-{index}", "prompt": prompt, "system": None, "expected_substring": expected})
        heldout_rows.append({"id": f"heldout-system-{index}", "prompt": prompt, "system": SYSTEMS[index % len(SYSTEMS)], "expected_substring": expected})
    write_jsonl(args.heldout_output, heldout_rows)
    print(json.dumps({"train_conversations": len(rows), "heldout_prompts": len(heldout_rows)}, indent=2))


if __name__ == "__main__":
    main()
