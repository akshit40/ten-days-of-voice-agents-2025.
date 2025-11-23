# backend/src/wellness_manager.py
import json
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "wellness_log.json")
LOG_PATH = os.path.abspath(LOG_PATH)

DEFAULT_ORDER = {
    "mood": "",
    "energy": "",
    "stressors": "",
    "objectives": [],
    "selfcare": "",
    "summary": ""
}

class WellnessManager:
    """
    Small session-level slot filling for a wellness check-in.
    The persistent log is stored as backend/wellness_log.json.
    """

    QUESTIONS = [
        ("mood", "How are you feeling today, in a few words?"),
        ("energy", "What's your energy like right now — low, medium, or high?"),
        ("stressors", "Anything stressing you out at the moment?"),
        ("objectives", "What are 1 to 3 things you'd like to get done today?"),
        ("selfcare", "Is there anything you want to do for yourself today (rest, short walk, hobby)?"),
        ("confirm", "Does this sound right? Say yes to save or say a change.")
    ]

    def __init__(self):
        self.entry = DEFAULT_ORDER.copy()
        self._asked_index = 0

    # naive parsing to capture quick user replies
    def update_from_text(self, text: str):
        if not text:
            return
        t = text.lower().strip()

        # if user says yes/no to confirm
        if self._asked_index > 0 and t in ("yes", "yeah", "yep", "correct", "right", "save"):
            # do nothing special — caller will check is_complete -> save
            return

        # assign to current question if missing
        if self._asked_index < len(self.QUESTIONS) - 1:
            key = self.QUESTIONS[self._asked_index][0]
            if key == "objectives":
                # split into up to 3 objectives by commas or 'and'
                parts = [p.strip() for p in text.replace(" and ", ",").split(",") if p.strip()]
                self.entry["objectives"] = parts[:3]
            else:
                # store full reply for mood/energy/stressors/selfcare
                self.entry[key] = text.strip()

    def next_question(self):
        # find next unanswered question (skip confirm)
        while self._asked_index < len(self.QUESTIONS):
            key, q = self.QUESTIONS[self._asked_index]
            if key == "confirm":
                # confirm only after main fields are set
                if self.is_ready_to_confirm():
                    return q
                else:
                    self._asked_index = 0
                    # loop will continue asking missing fields
            else:
                # if field missing, ask
                if key == "objectives":
                    if not self.entry["objectives"]:
                        return q
                else:
                    if not self.entry.get(key):
                        return q
            self._asked_index += 1
        return None

    def is_ready_to_confirm(self) -> bool:
        # require mood + energy + name objectives (at least 1)
        return bool(self.entry["mood"] and self.entry["energy"] and (self.entry["objectives"] or self.entry["selfcare"]))

    def is_complete(self) -> bool:
        # treat confirm as final step (must be ready and asked confirm)
        return self.is_ready_to_confirm() and self._asked_index >= (len(self.QUESTIONS) - 1)

    def build_summary(self) -> str:
        # small single-sentence summary
        objs = ", ".join(self.entry["objectives"]) if self.entry["objectives"] else "no specific objectives"
        summary = f"Mood: {self.entry['mood']}. Energy: {self.entry['energy']}. Objectives: {objs}."
        self.entry["summary"] = summary
        return summary

    def save(self):
        # load existing log
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log = []
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except Exception:
                log = []

        entry = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "mood": self.entry.get("mood"),
            "energy": self.entry.get("energy"),
            "stressors": self.entry.get("stressors"),
            "objectives": self.entry.get("objectives"),
            "selfcare": self.entry.get("selfcare"),
            "summary": self.entry.get("summary", "")
        }
        log.append(entry)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        return LOG_PATH, entry

    @staticmethod
    def load_log():
        if not os.path.exists(LOG_PATH):
            return []
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @staticmethod
    def last_entry():
        log = WellnessManager.load_log()
        if not log:
            return None
        return log[-1]
