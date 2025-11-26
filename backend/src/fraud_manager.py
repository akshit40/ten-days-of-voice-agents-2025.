# backend/src/fraud_manager.py
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared-data", "fraud_cases.json"))

def _read_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_db(data):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_case_by_username(username: str) -> Optional[Dict[str, Any]]:
    if not username:
        return None
    username = username.strip().lower()
    data = _read_db()
    for entry in data:
        if entry.get("userName", "").strip().lower() == username:
            return entry
    return None

def find_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    data = _read_db()
    for entry in data:
        if entry.get("id") == case_id:
            return entry
    return None

def update_case(case_id: str, updates: Dict[str, Any]) -> Optional[str]:
    data = _read_db()
    changed = False
    for i, entry in enumerate(data):
        if entry.get("id") == case_id:
            entry.update(updates)
            entry.setdefault("last_updated", datetime.now().isoformat())
            data[i] = entry
            changed = True
            break
    if changed:
        _write_db(data)
        return case_id
    return None

def list_cases():
    return _read_db()
