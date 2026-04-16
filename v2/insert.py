import csv

INPUT_FILE = "metadata.csv"
OUTPUT_FILE = "metadata_clean.csv"
PREFIX = r"D:\Downloads\dev-clean\male-voice-file"

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)

    fieldnames = [
        "file_name",
        "file_path",
        "speaker_name",
        "file_size_bytes",
        "word_count",
        "duration_seconds"
    ]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    for row in reader:
        file_name = row["audio_file"].strip()

        writer.writerow({
            "file_name": file_name,
            "file_path": f"{PREFIX}\\{file_name}",
            "speaker_name": row["speaker_name"].strip() if row["speaker_name"] else None,
            "file_size_bytes": int(row["file_size_bytes"]) if row["file_size_bytes"] else None,
            "word_count": int(row["transcript_word_count"]) if row["transcript_word_count"] else None,
            "duration_seconds": float(row["duration_seconds"]) if row["duration_seconds"] else None,
        })

print("✅ Done: metadata_clean.csv")