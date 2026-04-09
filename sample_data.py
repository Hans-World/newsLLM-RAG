"""
Sample top 100 records from each cleaned data JSON file.
Handles both JSON array format and newline-delimited JSON (NDJSON).
"""
import json
import os
from pathlib import Path

SOURCE_DIR = Path("/home/wilson081/proj/pts_project/data/cleaned_data")
OUTPUT_DIR = Path("/home/hanyusu/newsLLM-RAG/notebooks/data/samples")
N = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_NAME_MAP = {
    "中央社_sample.json":            "中央社.json",
    "公視_sample.json":              "公視.json",
    "台語台_sample.json":            "台語台.json",
    "央廣-2021-20260212_sample.json": "央廣.json",
    "客語_sample.json":              "客語.json",
    "華視_sample.json":              "華視.json",
}


def sample_json_array(filepath: Path, n: int) -> list:
    """Efficiently read first n items from a JSON array without loading the whole file."""
    import ijson
    records = []
    with open(filepath, "rb") as f:
        for record in ijson.items(f, "item"):
            records.append(record)
            if len(records) >= n:
                break
    return records


def sample_ndjson(filepath: Path, n: int) -> list:
    """Read first n lines from a newline-delimited JSON file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= n:
                break
    return records


def detect_and_sample(filepath: Path, n: int) -> list:
    """Auto-detect format and sample n records."""
    with open(filepath, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()

    if first_char == "[":
        print(f"  Format: JSON array")
        try:
            return sample_json_array(filepath, n)
        except ImportError:
            # ijson not available, fallback: load and slice (only safe for smaller files)
            print("  Warning: ijson not installed, falling back to full load (slow for large files)")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data[:n]
    elif first_char == "{":
        print(f"  Format: NDJSON")
        return sample_ndjson(filepath, n)
    else:
        print(f"  Unknown format (first char: {repr(first_char)}), trying NDJSON")
        return sample_ndjson(filepath, n)


def main():
    json_files = list(SOURCE_DIR.glob("*.json"))
    print(f"Found {len(json_files)} files in {SOURCE_DIR}\n")

    for filepath in json_files:
        stem = filepath.stem
        raw_name = f"{stem}_sample.json"
        out_path = OUTPUT_DIR / SAMPLE_NAME_MAP.get(raw_name, raw_name)
        print(f"Processing: {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)")

        records = detect_and_sample(filepath, N)
        print(f"  Sampled {len(records)} records")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"  Saved -> {out_path}\n")

    print("Done. Sample files:")
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        print(f"  {f.name}  ({f.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()