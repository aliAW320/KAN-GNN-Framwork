"""Extract numeric reference results from the provided paper, preserving tables."""
import csv
import hashlib
import re
import subprocess
from pathlib import Path


def extract_paper(pdf_path, destination):
    content = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], check=True,
                             capture_output=True, text=True).stdout
    sections = [
        ("1", "Table 1: Comparison with the base", "Computational Complexity.",
         ["bbbp", "tox21", "toxcast", "sider", "hiv", "bace"], "roc_auc"),
        ("2", "Table 2: Comparison", "Table 3: Comparison",
         ["bbbp", "tox21", "toxcast", "sider", "hiv", "bace"], "roc_auc"),
        ("3", "Table 3: Comparison with standard", "Table 4: Comparison",
         ["lipophilicity", "freesolv", "esol", "qm7", "qm8", "qm9"], "mae"),
    ]
    rows = []
    for table, begin, end, datasets, metric in sections:
        start = content.index(begin)
        block = content[start:content.index(end, start)]
        for line in block.splitlines():
            matches = list(re.finditer(r"(\d+\.\d+)\s*±\s*(\d+\.\d+)|(?<!\S)-(?!\S)", line))
            if len(matches) != 6:
                continue
            model = line[:matches[0].start()].strip()
            for dataset, match in zip(datasets, matches):
                rows.append({"table": table, "dataset": dataset, "paper_model": model,
                             "metric": metric, "mean": match.group(1) or "",
                             "std": match.group(2) or "", "runs": 5,
                             "status": "reported" if match.group(1) else "out_of_memory",
                             "pdf_sha256": hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()})
    if {row["table"] for row in rows} != {"1", "2", "3"}:
        raise ValueError("Expected paper comparison tables were not found")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with Path(destination).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
