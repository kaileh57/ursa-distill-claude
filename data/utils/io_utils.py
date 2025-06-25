import json
import hashlib
from typing import Any, Dict, List

def jload(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def jdump(data: Any, file_path: str) -> None:
    """Save data as JSON"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def question_hash(question: str) -> str:
    """Generate hash for question deduplication"""
    return hashlib.md5(question.encode('utf-8')).hexdigest()