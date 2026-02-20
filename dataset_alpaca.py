# convert_all_datasets.py
from datasets import load_dataset
import json

output_file = "combined_qa.jsonl"
count = 0

with open(output_file, "w", encoding="utf-8") as f:
    # 1. AmbigQA (~12K)
    print("Loading ambig_qa...")
    for split in ["train", "validation"]:
        ds = load_dataset("sewon/ambig_qa", "light", split=split)
        for ex in ds:
            question = ex["question"]
            ann = ex["annotations"]

            qa_pairs = ann.get("qaPairs", [])
            for pair in qa_pairs:
                answers = pair.get("answer", [])
                for answer_list in answers:
                    if answer_list:
                        answer = answer_list[0] if isinstance(answer_list, list) else answer_list
                        obj = {
                            "context": [f"User: {question}"],
                            "response": f"Assistant: {answer}"
                        }
                        f.write(json.dumps(obj) + "\n")
                        count += 1

            direct_answers = ann.get("answer", [])
            for answer_list in direct_answers:
                if answer_list:
                    answer = answer_list[0] if isinstance(answer_list, list) else answer_list
                    obj = {
                        "context": [f"User: {question}"],
                        "response": f"Assistant: {answer}"
                    }
                    f.write(json.dumps(obj) + "\n")
                    count += 1

    print(f"After ambig_qa: {count} examples")

    # 2. WikiQA (~1K, only correct answers)
    print("Loading wiki_qa...")
    for split in ["train", "validation", "test"]:
        ds = load_dataset("microsoft/wiki_qa", split=split)
        for ex in ds:
            if ex["label"] == 1:
                obj = {
                    "context": [f"User: {ex['question']}"],
                    "response": f"Assistant: {ex['answer']}"
                }
                f.write(json.dumps(obj) + "\n")
                count += 1

    print(f"After wiki_qa: {count} examples")

    # 3. TruthfulQA (~3K with all correct answers)
    print("Loading truthful_qa...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    for ex in ds:
        question = ex["question"]
        obj = {
            "context": [f"User: {question}"],
            "response": f"Assistant: {ex['best_answer']}"
        }
        f.write(json.dumps(obj) + "\n")
        count += 1

        for ans in ex.get("correct_answers", []):
            if ans != ex["best_answer"]:
                obj = {
                    "context": [f"User: {question}"],
                    "response": f"Assistant: {ans}"
                }
                f.write(json.dumps(obj) + "\n")
                count += 1

    print(f"After truthful_qa: {count} examples")

    # 4. SQuAD (~87K)
    print("Loading squad...")
    ds = load_dataset("squad", split="train")
    for ex in ds:
        question = ex["question"]
        answer = ex["answers"]["text"][0]
        obj = {
            "context": [f"User: {question}"],
            "response": f"Assistant: {answer}"
        }
        f.write(json.dumps(obj) + "\n")
        count += 1

    print(f"After squad: {count} examples")

    # 5. Dolly-15K (~15K, conversational)
    print("Loading dolly...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in ds:
        instruction = ex["instruction"]
        context = ex.get("context", "")
        response = ex["response"]

        # Include context if present
        if context:
            user_text = f"User: {instruction}\n\nContext: {context}"
        else:
            user_text = f"User: {instruction}"

        obj = {
            "context": [user_text],
            "response": f"Assistant: {response}"
        }
        f.write(json.dumps(obj) + "\n")
        count += 1

    print(f"After dolly: {count} examples")

print(f"\n=== Total: {count} examples saved to {output_file} ===")