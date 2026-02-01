# backend/src/tutor_manager.py
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

HERE = os.path.dirname(__file__)
CONTENT_PATH = os.path.abspath(os.path.join(HERE, "..", "shared-data", "day4_tutor_content.json"))
STATE_PATH = os.path.abspath(os.path.join(HERE, "..", "tutor_state.json"))

def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _safe_write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

class TutorManager:
    """
    Enhanced tutor manager:
    - loads concept content
    - persists session mastery in tutor_state.json
    - richer mastery model per concept
    - heuristic teach-back evaluator (improved)
    """

    def __init__(self):
        content = _safe_load_json(CONTENT_PATH) or []
        # map by id
        self.content: Dict[str, Dict[str, Any]] = {c["id"]: c for c in content}
        # load global state (session_id -> state)
        saved = _safe_load_json(STATE_PATH) or {}
        self.state: Dict[str, Any] = saved
        # in-memory session control (mode, concept, last_question)
        self.sessions: Dict[str, Dict[str, Any]] = {}

    # -----------------------
    # Content access helpers
    # -----------------------
    def list_concepts(self) -> List[Dict[str, Any]]:
        return list(self.content.values())

    def choose_concept(self, key_or_keyword: str):
        if not key_or_keyword:
            return next(iter(self.content.values()), None)
        if key_or_keyword in self.content:
            return self.content[key_or_keyword]
        k = key_or_keyword.lower()
        for c in self.content.values():
            if k in c.get("title","").lower() or k in c.get("summary","").lower():
                return c
        return None

    def get_concept(self, concept_id: str):
        return self.content.get(concept_id)

    # -----------------------
    # Session management
    # -----------------------
    def start_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "mode": None,
                "current_concept": None,
                "last_question": None
            }
        return self.sessions[session_id]

    def set_mode(self, session_id: str, mode: str, concept_id: Optional[str] = None):
        sess = self.start_session(session_id)
        sess["mode"] = mode
        if concept_id:
            resolved = concept_id if concept_id in self.content else (self.choose_concept(concept_id) or {}).get("id")
            sess["current_concept"] = resolved
        if not sess.get("current_concept"):
            first = next(iter(self.content.values()), None)
            sess["current_concept"] = first["id"] if first else None
        return sess

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    # -----------------------
    # Mastery model helpers
    # -----------------------
    def _ensure_session_state(self, session_id: str):
        s = self.state.get(session_id)
        if not s:
            s = {
                "mastery": {},  # concept_id -> metrics
                "updated_at": datetime.now().astimezone().isoformat()
            }
            self.state[session_id] = s
        return s

    def _ensure_concept_entry(self, session_id: str, concept_id: str):
        master = self._ensure_session_state(session_id)
        mastery = master.setdefault("mastery", {})
        entry = mastery.get(concept_id)
        if not entry:
            entry = {
                "times_explained": 0,
                "times_quizzed": 0,
                "times_taught_back": 0,
                "last_score": None,
                "avg_score": None
            }
            mastery[concept_id] = entry
        return entry

    def record_explain(self, session_id: str, concept_id: str):
        e = self._ensure_concept_entry(session_id, concept_id)
        e["times_explained"] = e.get("times_explained", 0) + 1
        self.state[session_id]["updated_at"] = datetime.now().astimezone().isoformat()
        _safe_write_json(STATE_PATH, self.state)

    def record_quiz_result(self, session_id: str, concept_id: str, correct: bool):
        e = self._ensure_concept_entry(session_id, concept_id)
        e["times_quizzed"] = e.get("times_quizzed", 0) + 1
        score = 100 if correct else 0
        e["last_score"] = score
        if e.get("avg_score") is None:
            e["avg_score"] = score
        else:
            # running average
            prev = e["avg_score"]
            e["avg_score"] = round((prev * (e["times_quizzed"] - 1) + score) / e["times_quizzed"], 1)
        self.state[session_id]["updated_at"] = datetime.now().astimezone().isoformat()
        _safe_write_json(STATE_PATH, self.state)

    def record_taught_back(self, session_id: str, concept_id: str, score: int):
        e = self._ensure_concept_entry(session_id, concept_id)
        e["times_taught_back"] = e.get("times_taught_back", 0) + 1
        e["last_score"] = score
        if e.get("avg_score") is None:
            e["avg_score"] = score
        else:
            # running average across teach_back + quizzes combined is fine:
            # keep an exponential-ish smoothing: new_avg = round((prev + score)/2,1)
            prev = e.get("avg_score", score)
            e["avg_score"] = round((prev + score) / 2, 1)
        self.state[session_id]["updated_at"] = datetime.now().astimezone().isoformat()
        _safe_write_json(STATE_PATH, self.state)

    def get_mastery(self, session_id: str) -> Dict[str, Any]:
        return self.state.get(session_id, {}).get("mastery", {})

    def get_weakest(self, session_id: str, top_n: int = 3) -> List[Tuple[str, float]]:
        mastery = self.get_mastery(session_id)
        arr = []
        for k, v in mastery.items():
            avg = v.get("avg_score") or 0
            arr.append((k, avg))
        arr.sort(key=lambda x: x[1])  # ascending -> weakest first
        return arr[:top_n]

    # -----------------------
    # Quiz / teach-back prompts
    # -----------------------
    def ask_quiz_question(self, session_id: str) -> Optional[str]:
        sess = self.start_session(session_id)
        cid = sess.get("current_concept")
        c = self.get_concept(cid)
        if not c:
            return None
        q = c.get("sample_question", "Explain the concept in your own words.")
        sess["last_question"] = q
        return q

    def ask_teach_back_prompt(self, session_id: str) -> Optional[str]:
        sess = self.start_session(session_id)
        cid = sess.get("current_concept")
        c = self.get_concept(cid)
        if not c:
            return None
        p = f"Please explain {c.get('title')} in your own words."
        sess["last_question"] = p
        return p

    # -----------------------
    # Evaluator: improved heuristic
    # -----------------------
    def _tokenize(self, s: str):
        if not s:
            return []
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        tokens = [t for t in s.split() if len(t) > 1]
        return tokens

    def _important_tokens(self, summary: str):
        # pick nouns / core tokens heuristically: remove common stopwords and short words
        stop = {
            "the","a","an","and","or","to","of","in","on","for","with","that","is","it","so",
            "you","can","be","this","by","as","are","from","we","will","they","their"
        }
        tokens = [t for t in self._tokenize(summary) if t not in stop]
        return tokens

    def evaluate_teach_back(self, concept_summary: str, user_text: str) -> Dict[str, Any]:
        """
        Returns:
          { score: int (0-100),
            feedback: str,
            signals: {overlap_count, important_hit_count, example_found: bool} }
        Heuristic:
          - compute overlap ratio with summary tokens (base)
          - check presence of 'example' keywords or numeric/example phrases (+ bonus)
          - reward hitting important tokens (title nouns etc.)
        """
        sum_tokens = self._tokenize(concept_summary)
        user_tokens = self._tokenize(user_text)

        sum_set = set(sum_tokens)
        user_set = set(user_tokens)

        if not sum_tokens:
            return {"score": 0, "feedback": "No summary available to evaluate.", "signals": {}}

        overlap = sum_set.intersection(user_set)
        overlap_ratio = len(overlap) / max(1, len(sum_set))

        # important tokens (usually nouns/keywords)
        important = set(self._important_tokens(concept_summary))
        important_hits = len(important.intersection(user_set))
        important_ratio = important_hits / max(1, len(important)) if important else 0.0

        # detect if user provided an example phrase (very naive)
        example_keywords = ("for example", "for instance", "like", "such as", "e.g.", "example")
        example_found = any(k in user_text.lower() for k in example_keywords)

        # base score: weighted mix
        score = (overlap_ratio * 0.6 + important_ratio * 0.35 + (0.05 if example_found else 0.0)) * 100
        score = int(max(0, min(100, round(score))))

        # feedback messages
        if score >= 85:
            feedback = "Great explanation — you covered most key points and even provided examples."
        elif score >= 60:
            feedback = "Good explanation. You hit several important points; adding a short example would make it stronger."
        elif score >= 35:
            feedback = "You mentioned some relevant ideas but missed key details. Try focusing on the main purpose and one simple example."
        else:
            feedback = "This missed several main ideas. Try explaining the core purpose and one simple example."

        return {
            "score": score,
            "feedback": feedback,
            "signals": {
                "overlap_count": len(overlap),
                "important_hit_count": important_hits,
                "example_found": bool(example_found),
            },
        }

    # convenience: save state explicitly if modified externally
    def save_state(self):
        _safe_write_json(STATE_PATH, self.state)

# end of tutor_manager.py
