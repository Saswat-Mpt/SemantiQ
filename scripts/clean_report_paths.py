from __future__ import annotations

import json
from pathlib import Path

# Fix relative filepaths in reports
PROJECT_ROOT = Path(__file__).resolve().parents[1]

for report_file in (PROJECT_ROOT / "reports").glob("*.json"):
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace absolute Windows user paths with relative clean paths
        if "C:\\Users\\hp\\Downloads\\NLP Project\\SemantiQ\\" in content or "C:/Users/hp/Downloads/NLP Project/SemantiQ/" in content:
            cleaned = content.replace("C:\\\\Users\\\\hp\\\\Downloads\\\\NLP Project\\\\SemantiQ\\\\", "")
            cleaned = cleaned.replace("C:/Users/hp/Downloads/NLP Project/SemantiQ/", "")
            cleaned = cleaned.replace("C:\\Users\\hp\\Downloads\\NLP Project\\SemantiQ\\", "")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(cleaned)
    except Exception:
        pass
