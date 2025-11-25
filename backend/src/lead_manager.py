import json
import os
from datetime import datetime

def new_lead_template():
    return {
        "name": None,
        "company": None,
        "email": None,
        "role": None,
        "use_case": None,
        "team_size": None,
        "timeline": None,
        "notes": None,
        "collected_at": None
    }

def save_lead(lead: dict, folder: str = None) -> str:
    folder = folder or os.path.join(os.path.dirname(__file__), "..", "leads")
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    filename = f"lead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lead, f, indent=2, ensure_ascii=False)
    return path

def build_summary(lead: dict) -> str:
    parts = []
    if lead.get("name"):
        parts.append(f"{lead['name']} from {lead.get('company','(company unknown)')}")
    else:
        parts.append(f"{lead.get('company','(company unknown)')}")
    if lead.get("role"):
        parts.append(f"role: {lead['role']}")
    if lead.get("use_case"):
        parts.append(f"use: {lead['use_case']}")
    if lead.get("team_size"):
        parts.append(f"team: {lead['team_size']}")
    if lead.get("timeline"):
        parts.append(f"timeline: {lead['timeline']}")
    if lead.get("email"):
        parts.append(f"email: {lead['email']}")
    return "; ".join(parts)
