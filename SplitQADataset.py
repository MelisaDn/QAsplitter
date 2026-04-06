import json
from Corpus.TurkishSplitter import TurkishSplitter

input_file = "QAdataset/tr-dev-v1.1.json"
output_file = "QAdataset/splitted-tr-dev-v1.1.json"


def normalize_text(text: str) -> str:
    """
    Normalizes special characters without removing important information.
    """
    if text is None:
        return ""

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
    Splits a text into sentences using TurkishSplitter.
    """
    text = normalize_text(text).strip()
    if not text:
        return []

    sentences = splitter.split(text)
    return [
        sentence.toString().strip()
        for sentence in sentences
        if sentence.toString().strip()
    ]


splitter = TurkishSplitter()

with open(input_file, "r", encoding="utf-8") as file:
    dataset = json.load(file)

for article in dataset.get("data", []):
    article["title"] = normalize_text(article.get("title", ""))

    for paragraph in article.get("paragraphs", []):
        normalized_context = normalize_text(paragraph.get("context", ""))

        # Keep original context
        paragraph["context"] = normalized_context

        # Add sentence-splitted context
        paragraph["context_sentences"] = split_text_into_sentences(
            normalized_context, splitter
        )

        for qa in paragraph.get("qas", []):
            normalized_question = normalize_text(qa.get("question", ""))

            # Keep original question (no sentence split needed)
            qa["question"] = normalized_question

            # Normalize answers only
            for answer in qa.get("answers", []):
                if "text" in answer and isinstance(answer["text"], str):
                    answer["text"] = normalize_text(answer["text"])

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(dataset, file, ensure_ascii=False, indent=2)

print(f"File saved: {output_file}")