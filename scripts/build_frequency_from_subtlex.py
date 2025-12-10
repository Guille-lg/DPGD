import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd


def build_frequency_csv_from_subtlex(
    excel_path: Path,
    output_csv: Path,
    sheet_name=0,
) -> None:
    """Parse SUBTLEX-ESP Excel and build a unified word,count CSV.

    The SUBTLEX-ESP sheet repeats blocks of columns like:
        [Word, Freq. count, Freq. per million, Log freq.]
    across the sheet. This function:
      - finds every such block whose first column header contains "word"
        (case-insensitive) and whose second column header contains
        "freq" and "count";
      - collects (word, freq_count) rows from all blocks;
      - aggregates counts per word (summing across blocks if repeated);
      - writes a CSV with header `word,count` sorted by descending count.
    """

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    print(f"Loading SUBTLEX-ESP from: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Normalize column names to strings
    cols = [str(c) for c in df.columns]

    # Identify blocks: we look for indices i such that
    #   cols[i]   ~ word
    #   cols[i+1] ~ freq count
    block_indices = []
    for i in range(len(cols) - 1):
        c0 = cols[i].lower()
        c1 = cols[i + 1].lower()
        if "word" in c0 and "freq" in c1 and "count" in c1:
            block_indices.append((i, i + 1))

    if not block_indices:
        raise ValueError("No [Word, Freq. count] column blocks found in the sheet.")

    print(f"Found {len(block_indices)} word/frequency column block(s): {block_indices}")

    freqs = defaultdict(int)

    for word_idx, count_idx in block_indices:
        word_col = df.columns[word_idx]
        count_col = df.columns[count_idx]
        print(f"Processing block: word_col='{word_col}', count_col='{count_col}'")

        for _, row in df[[word_col, count_col]].dropna(subset=[word_col, count_col]).iterrows():
            word = str(row[word_col]).strip()
            # skip empty or non-string-like words
            if not word:
                continue

            try:
                count = int(row[count_col])
            except (ValueError, TypeError):
                # skip rows with non-integer counts
                continue

            if count < 0:
                continue

            freqs[word] += count

    if not freqs:
        raise ValueError("No frequency entries extracted from SUBTLEX-ESP.")

    # Sort by descending count
    items = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing unified frequency CSV to: {output_csv}")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count"])
        for word, count in items:
            writer.writerow([word, count])

    print(f"Wrote {len(items)} word entries.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frequency_es.csv from SUBTLEX-ESP Excel file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/SUBTLEX-ESP.xlsx",
        help="Path to SUBTLEX-ESP Excel file (default: data/SUBTLEX-ESP.xlsx)",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="0",
        help="Sheet name or index (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/frequency_es.csv",
        help="Output CSV path (default: data/frequency_es.csv)",
    )

    args = parser.parse_args()

    # Interpret sheet argument
    try:
        sheet_name: object = int(args.sheet)
    except ValueError:
        sheet_name = args.sheet

    excel_path = Path(args.input)
    output_csv = Path(args.output)

    build_frequency_csv_from_subtlex(excel_path, output_csv, sheet_name=sheet_name)


if __name__ == "__main__":
    main()
