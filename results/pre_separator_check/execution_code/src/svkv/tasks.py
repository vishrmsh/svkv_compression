"""Public LongBench subsets and an explicitly non-official NIAH stress test."""
import hashlib
import json
import random
import re
import string
import zipfile
from collections import Counter


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def split_prompt(tokenizer, context, question, instruction):
    marker = "<<<QUESTION_BOUNDARY>>>"
    rendered = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction + "\n\n" + context + "\n\n" + marker + question},
    ], tokenize=False, add_generation_prompt=True)
    prefix, suffix = rendered.split(marker)
    return tokenizer.encode(prefix, add_special_tokens=False), tokenizer.encode(suffix, add_special_tokens=False)


def build_cases(tokenizer, archive, lengths=(8192, 16384, 32768), seeds=(17, 29), depths=(.1, .5, .9), longbench_n=6):
    z = zipfile.ZipFile(archive)
    source = lambda task: [json.loads(x) for x in z.read(f"data/{task}.jsonl").splitlines()]
    # Natural prose distractors; these are not official RULER instances.
    filler = "\n\n".join(x["context"] for x in source("qasper")[:3])
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    cases = []
    instruction = "Read the following text and remember the special magic numbers. You will be asked about them afterwards."
    for length in lengths:
        for kind in ("niah_single", "niah_multi"):
            for seed in seeds:
                for depth in depths:
                    rng = random.Random(seed + int(depth * 1000) + length + (kind == "niah_multi") * 100000)
                    names = ["cobalt otter", "amber kestrel"][:1 if kind == "niah_single" else 2]
                    answers = [str(rng.randint(1000000, 9999999)) for _ in names]
                    question = "What are the special magic numbers for " + " and ".join(names) + "? Reply only with the numbers."
                    header, suffix = split_prompt(tokenizer, "", question, instruction)
                    needle_ids = [tokenizer.encode(f"\nOne of the special magic numbers for {name} is: {answer}.\n", add_special_tokens=False) for name, answer in zip(names, answers)]
                    # 256-token reserve keeps all input + generation within the advertised window.
                    size = length - 256 - len(header) - sum(map(len, needle_ids))
                    offset = rng.randrange(len(filler_ids))
                    doubled = filler_ids[offset:] + filler_ids[:offset]
                    filler_tokens = (doubled * (size // len(doubled) + 1))[:size]
                    positions = [int(size * depth)]
                    if len(names) == 2:
                        positions.append(int(size * (.85 if depth < .5 else .15)))
                    inserts = sorted(zip(positions, needle_ids))
                    tokens, spans, prev = list(header), [], 0
                    for pos, needle in inserts:
                        tokens.extend(filler_tokens[prev:pos]); start = len(tokens)
                        tokens.extend(needle); spans.append([start, len(tokens)]); prev = pos
                    tokens.extend(filler_tokens[prev:])
                    case_id = f"{kind}_{length}_s{seed}_d{depth}"
                    cases.append(dict(id=case_id, task=kind, nominal_length=length, seed=seed, depth=depth,
                                      prefix=tokens, suffix=suffix, answers=answers, needle_spans=spans,
                                      prefix_sha256=digest(tokens), source="custom_niah_natural_prose"))
    instructions = {
        "hotpotqa": "Answer the question based on the given passages. Give only the answer, without any explanation.",
        "qasper": "Answer the question based on the scientific article below. Give a concise answer only. If the article does not answer the question, say unanswerable.",
        "passage_retrieval_en": "Find the paragraph most relevant to the summary that follows the paragraphs. Answer only with Paragraph followed by its number.",
    }
    for task, instruction in instructions.items():
        pool = source(task)
        order = list(range(len(pool)))
        random.Random(20260927).shuffle(order)
        accepted = 0
        for row_index in order:
            row = pool[row_index]
            question = ("Summary: " if task == "passage_retrieval_en" else "Question: ") + row["input"] + "\nAnswer:"
            prefix, suffix = split_prompt(tokenizer, row["context"], question, instruction)
            # Select naturally long inputs without truncating away gold evidence.
            if not (7900 <= len(prefix) <= 32000 and len(prefix) + len(suffix) + 128 <= 32768):
                continue
            cases.append(dict(id=f"longbench_{task}_{row_index}", task=task, nominal_length=len(prefix),
                              seed=20260927, prefix=prefix, suffix=suffix, answers=row["answers"],
                              needle_spans=[], prefix_sha256=digest(prefix), source="THUDM/LongBench",
                              source_row=row_index, source_id=row.get("_id")))
            accepted += 1
            if accepted >= longbench_n:
                break
        if accepted != longbench_n:
            raise ValueError(f"Only {accepted} eligible rows for {task}")
    return cases


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score_answer(task, prediction, answers):
    if task.startswith("niah"):
        return sum(bool(re.search(r"(?<!\d)" + re.escape(a) + r"(?!\d)", prediction)) for a in answers) / len(answers)
    if task == "passage_retrieval_en":
        # LongBench retrieval metric: credit fraction of predicted numbers equal to gold.
        numbers = re.findall(r"\d+", prediction)
        return max((sum(n == re.search(r"\d+", a).group() for n in numbers) / len(numbers) if numbers else 0) for a in answers)
    pred = normalize(prediction).split()
    scores = []
    for answer in answers:
        gold = normalize(answer).split()
        common = sum((Counter(pred) & Counter(gold)).values())
        scores.append(2 * common / (len(pred) + len(gold)) if pred or gold else 1.)
    return max(scores, default=0.)
