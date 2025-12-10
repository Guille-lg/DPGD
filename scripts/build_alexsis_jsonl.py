import json
from collections import Counter
from pathlib import Path


def choose_mode_substitution(subs):
    """Choose the most frequent non-empty substitution (mode).

    Normalizes by stripping whitespace. If no valid substitutions, returns None.
    """
    cleaned = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    if not cleaned:
        return None
    counter = Counter(cleaned)
    # most_common returns list of (item, count) sorted by count desc
    return counter.most_common(1)[0][0]


def replace_first(target_sentence: str, complex_word: str, substitution: str) -> str:
    """Replace the first occurrence of complex_word in target_sentence with substitution.

    This is a simple string-level replacement, matching the ALEXSIS description
    (first occurrence marked for annotators). If the complex word is not found,
    the original sentence is returned unchanged.
    """
    complex_word = complex_word.strip()
    if not complex_word:
        return target_sentence
    return target_sentence.replace(complex_word, substitution, 1)


def build_alexsis_jsonl(
    tsv_path: Path,
    output_path: Path,
    encoding: str = "utf-8",
) -> None:
    """Convert ALEXSIS_v1.0.tsv to data/alexsis.jsonl with source/reference pairs.

    Each TSV line is expected to have the format:
        SENTENCE<TAB>COMPLEX_WORD<TAB>SUB_1<TAB>...<TAB>SUB_25

    The script:
      - uses SENTENCE as `source`
      - selects the mode (most frequent) candidate among SUB_1..SUB_25
      - creates `reference` by replacing the first occurrence of COMPLEX_WORD
        in SENTENCE with that chosen substitution
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_lines = 0
    num_written = 0
    num_skipped_no_sub = 0

    with tsv_path.open("r", encoding=encoding, newline="") as fin, \
            output_path.open("w", encoding="utf-8", newline="") as fout:
        for line in fin:
            line = line.rstrip("\n\r")
            if not line:
                continue

            num_lines += 1
            parts = line.split("\t")
            if len(parts) < 3:
                # Not enough fields; skip
                continue

            sentence = parts[0].strip()
            complex_word = parts[1].strip()
            substitutions = parts[2:]

            best_sub = choose_mode_substitution(substitutions)
            if best_sub is None:
                num_skipped_no_sub += 1
                continue

            reference = replace_first(sentence, complex_word, best_sub)

            record = {
                "source": sentence,
                "reference": reference,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Processed lines: {num_lines}")
    print(f"Written JSONL records: {num_written}")
    if num_skipped_no_sub:
        print(f"Skipped lines with no valid substitutions: {num_skipped_no_sub}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    tsv_file = project_root / "data" / "ALEXSIS_v1.0.tsv"
    output_file = project_root / "data" / "alexsis.jsonl"
    build_alexsis_jsonl(tsv_file, output_file)
