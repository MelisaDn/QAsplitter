import json
import re
from Corpus.TurkishSplitter import TurkishSplitter

input_file = "QAdataset/tr-train-v1.1.json"
output_file = "QAdataset/splitted-tr-train-v1.1.json"


def normalize_text(text: str) -> str:
    """
    Normalizes special characters without removing important information.
    """
    return (
        text.replace("□", "–")
            .replace("—", "–")
            .replace("’", "'")
            .replace("‒", "–")
            .replace("<<", "«")
            .replace(">>", "»")
            .replace("•", "–")
    )


def split_text_into_sentences(text: str, splitter: TurkishSplitter) -> list[str]:
    """
    Splits a text into sentences using TurkishSplitter and returns them as strings.
    """
    if text is None:
        return []

    text = normalize_text(text).strip()
    if not text:
        return []

    sentences = splitter.split(text)
    return [sentence.toString().strip() for sentence in sentences if sentence.toString().strip()]


def find_sentence_offsets(original_text: str, split_sentences: list[str]) -> list[dict]:
    """
    Tries to find sentence start/end character offsets inside the original text.

    This is useful later if you want to connect sentence-level parsing results
    back to the original paragraph without losing QA answer positions.

    Returns:
        [
            {
                "sentence": "...",
                "start_char": ...,
                "end_char": ...
            },
            ...
        ]
    """
    offsets = []
    cursor = 0

    for sentence in split_sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue

        start_index = original_text.find(sentence_clean, cursor)

        if start_index == -1:
            # fallback: try space-normalized matching
            original_slice = original_text[cursor:]
            normalized_original = re.sub(r"\s+", " ", original_slice)
            normalized_sentence = re.sub(r"\s+", " ", sentence_clean)

            approx_index = normalized_original.find(normalized_sentence)
            if approx_index == -1:
                offsets.append({
                    "sentence": sentence_clean,
                    "start_char": None,
                    "end_char": None
                })
                continue
            else:
                # If normalized match succeeds but exact position is hard to recover safely,
                # keep sentence and leave offsets unknown rather than writing wrong offsets.
                offsets.append({
                    "sentence": sentence_clean,
                    "start_char": None,
                    "end_char": None
                })
                continue

        end_index = start_index + len(sentence_clean)

        offsets.append({
            "sentence": sentence_clean,
            "start_char": start_index,
            "end_char": end_index
        })

        cursor = end_index

    return offsets


splitter = TurkishSplitter()

with open(input_file, "r", encoding="utf-8") as file:
    dataset = json.load(file)

for article in dataset.get("data", []):
    article["title"] = normalize_text(article.get("title", ""))

    for paragraph in article.get("paragraphs", []):
        original_context = paragraph.get("context", "")
        normalized_context = normalize_text(original_context)

        # Keep original context for QA
        paragraph["context"] = normalized_context

        # Add sentence-splitted context for later stanza/dependency parsing
        context_sentences = split_text_into_sentences(normalized_context, splitter)
        paragraph["context_sentences"] = context_sentences

        # Optional: keep sentence offsets relative to the original context
        paragraph["context_sentence_offsets"] = find_sentence_offsets(
            normalized_context,
            context_sentences
        )

        for qa in paragraph.get("qas", []):
            original_question = qa.get("question", "")
            normalized_question = normalize_text(original_question)

            # Keep original question text
            qa["question"] = normalized_question

            # Add split version for later parsing
            qa["question_sentences"] = split_text_into_sentences(normalized_question, splitter)

            # Keep answers untouched, only normalize text field
            for answer in qa.get("answers", []):
                if "text" in answer and isinstance(answer["text"], str):
                    answer["text"] = normalize_text(answer["text"])

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(dataset, file, ensure_ascii=False, indent=2)

print(f"File saved: {output_file}")