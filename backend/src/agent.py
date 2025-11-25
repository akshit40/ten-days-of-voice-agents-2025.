# backend/src/agent.py
import logging
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# existing managers (order + wellness) and new tutor manager
from wellness_manager import WellnessManager
from tutor_manager import TutorManager

# lead manager for SDR (Day 5)
from lead_manager import new_lead_template, save_lead, build_summary

logger = logging.getLogger("agent")
load_dotenv(".env.local")


#
# ----------------------- OrderManager (unchanged) -----------------------
#
class OrderManager:
    def __init__(self):
        self.order = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": "",
        }

    def update_from_text(self, text: str):
        if not text:
            return
        t = text.lower()
        for d in ["latte", "cappuccino", "americano", "espresso", "mocha", "cold brew", "flat white"]:
            if d in t:
                self.order["drinkType"] = d
        for s in ["small", "medium", "large"]:
            if s in t:
                self.order["size"] = s
        for m in ["whole", "skim", "oat", "soy", "almond", "2%"]:
            if m in t:
                self.order["milk"] = m
        for ex in ["vanilla", "caramel", "hazelnut", "whipped", "extra shot", "shot"]:
            if ex in t and ex not in self.order["extras"]:
                self.order["extras"].append(ex)
        if "my name is " in t:
            try:
                name = t.split("my name is ", 1)[1].strip().split()[0]
                self.order["name"] = name.capitalize()
            except Exception:
                pass
        elif " for " in t:
            try:
                name = t.split(" for ", 1)[1].strip().split()[0]
                self.order["name"] = name.capitalize()
            except Exception:
                pass

    def is_complete(self) -> bool:
        return bool(self.order["drinkType"] and self.order["size"] and self.order["milk"] and self.order["name"])

    def next_question(self) -> str | None:
        if not self.order["drinkType"]:
            return "What would you like to drink today? We have latte, cappuccino, americano, mocha, and espresso."
        if not self.order["size"]:
            return "What size would you like — small, medium, or large?"
        if not self.order["milk"]:
            return "Which milk would you prefer — whole, skim, oat, soy or almond?"
        if not self.order["extras"]:
            return "Any extras — caramel, vanilla, whipped cream, or an extra shot?"
        if not self.order["name"]:
            return "Under what name should I put this order?"
        return None

    def save(self, folder: str = "orders") -> str:
        os.makedirs(folder, exist_ok=True)
        filename = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.order, f, indent=2, ensure_ascii=False)
        return path


#
# ----------------------- Agent / Entry point -----------------------
#
class Assistant(Agent):
    def __init__(self) -> None:
        # SDR persona + multi-role instructions:
        super().__init__(
            instructions="""
You are a helpful voice AI assistant that can act in three roles: SDR (Sales Development Rep), wellness check-in companion, and an active-recall tutor.
When the user expresses interest in product, pricing, demo, or says they're 'interested', prioritize acting as the SDR.

SRD role rules (strict):
- Greet warmly, be concise, and ask short questions.
- Use ONLY the provided FAQ JSON (shared-data/company_faq_*.json) to answer product/pricing questions. If the FAQ doesn't contain the info, say: "I don't have that detail — would you like me to book a demo or pass this to our product team?"
- Goal: understand the user's problem and collect lead fields: name, company, email, role, use_case, team_size, timeline.
- Ask clarifying questions naturally to collect missing fields.
- When user says "save" or "that's all" or "thanks", save the lead JSON and read a short summary + path.
- Do NOT ask for or store sensitive information (cards, IDs, health data).
- Keep responses short (1-2 sentences) for TTS clarity.

Wellness & Tutor roles:
- If user requests a daily check-in, act as the wellness companion and persist entries to JSON.
- If user requests tutoring, support learn / quiz / teach_back flows and track simple mastery.

Keep language friendly, practical, and non-diagnostic. Keep answers concise for TTS.
"""
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# --- helper: robust FAQ matching using token overlap
def best_faq_for_text(user_text: str, faq_list: list[dict]) -> tuple[dict | None, int]:
    """
    Return (best_faq_entry, score). Score is simple token overlap of question + important answer tokens.
    """
    if not user_text or not faq_list:
        return None, 0
    t = user_text.lower()
    best = None
    best_score = 0
    for f in faq_list:
        q = (f.get("question") or "").lower()
        a = (f.get("answer") or "").lower()
        # tokens from question and leading answer tokens
        tokens = [tok for tok in re.split(r"\W+", q) if tok]
        tokens += [tok for tok in re.split(r"\W+", a) if tok][:6]
        # compute overlap score
        score = sum(1 for tok in set(tokens) if tok and tok in t)
        if score > best_score:
            best_score = score
            best = f
    return best, best_score


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # managers
    order_managers: dict[str, OrderManager] = {}
    wellness_managers: dict[str, WellnessManager] = {}
    tutor = TutorManager()
    tutor_sessions = {}  # optional per-session quick state (mirrors tutor.sessions)

    def get_mgr(session_id: str) -> OrderManager:
        if session_id not in order_managers:
            order_managers[session_id] = OrderManager()
        return order_managers[session_id]

    def get_wellness_mgr(session_id: str) -> WellnessManager:
        if session_id not in wellness_managers:
            wellness_managers[session_id] = WellnessManager()
        return wellness_managers[session_id]

    # ---------------------------
    # respond factory - supports per-message voice override
    # ---------------------------
    def respond_fn_factory(sess, ctx_obj):
        async def respond_fn(reply_text: str, voice: str | None = None):
            """
            Send a reply. Prefer structured payload with tts override (some SDKs accept JSON).
            Falls back to simple text sends and room data.
            """
            sent = False
            # Build structured payload if voice requested
            if voice:
                payload = {"text": reply_text, "tts": {"voice": voice}}
                try:
                    if hasattr(sess, "send_text") and callable(sess.send_text):
                        await sess.send_text(payload)
                        logger.debug("Sent structured payload via send_text with voice override: %s", voice)
                        return
                except Exception as e:
                    logger.debug("structured send_text failed: %s", e)

                try:
                    if hasattr(sess, "publish_text") and callable(sess.publish_text):
                        await sess.publish_text(payload)
                        logger.debug("Sent structured payload via publish_text with voice override: %s", voice)
                        return
                except Exception as e:
                    logger.debug("structured publish_text failed: %s", e)

            # Fallback to plain text message
            try:
                if hasattr(sess, "send_text") and callable(sess.send_text):
                    await sess.send_text(reply_text)
                    sent = True
            except Exception:
                logger.debug("session.send_text not available or failed")

            if not sent:
                try:
                    if hasattr(sess, "publish_text") and callable(sess.publish_text):
                        await sess.publish_text(reply_text)
                        sent = True
                except Exception:
                    logger.debug("session.publish_text not available or failed")

            if not sent:
                try:
                    agent_obj = getattr(sess, "agent", None)
                    if agent_obj and hasattr(agent_obj, "send_message"):
                        await agent_obj.send_message(reply_text)
                        sent = True
                except Exception:
                    logger.debug("agent.send_message not available or failed")

            if not sent:
                try:
                    await ctx_obj.room.send_data(reply_text)
                    sent = True
                except Exception:
                    logger.exception("Failed to send reply via any available method")

        return respond_fn

    # coffee handler (unchanged, short)
    async def handle_coffee(session_id: str, user_text: str, respond_fn):
        try:
            mgr = get_mgr(session_id)
            mgr.update_from_text(user_text)
            if mgr.is_complete():
                path = mgr.save()
                summary_text = f"Perfect — your order is a {mgr.order['size']} {mgr.order['drinkType']} with {mgr.order['milk']} milk"
                if mgr.order["extras"]:
                    summary_text += f" and extras: {', '.join(mgr.order['extras'])}"
                summary_text += f" for {mgr.order['name']}. I've saved it to {path}. Enjoy!"
                await respond_fn(summary_text)
                try:
                    del order_managers[session_id]
                except Exception:
                    pass
                return
            q = mgr.next_question()
            if q:
                await respond_fn(q)
                return
            await respond_fn("Sorry, I didn't catch that. Could you repeat please?")
        except Exception as e:
            logger.exception("Error in handle_coffee: %s", e)
            try:
                await respond_fn("Something went wrong processing your order.")
            except Exception:
                pass

    # wellness handler (unchanged)
    async def handle_wellness(session_id: str, user_text: str, respond_fn):
        try:
            mgr = get_wellness_mgr(session_id)
            mgr.update_from_text(user_text)
            q = mgr.next_question()
            if q:
                await respond_fn(q)
                mgr._asked_index += 1
                return
            if mgr.is_ready_to_confirm() and not mgr.is_complete():
                summary = mgr.build_summary()
                await respond_fn(f"Quick summary: {summary} Do you want me to save this check-in?")
                mgr._asked_index = len(mgr.QUESTIONS) - 1
                return
            if mgr.is_complete():
                path, saved_entry = mgr.save()
                await respond_fn(f"Saved today's check-in. Summary: {saved_entry.get('summary','')}. I saved it to {path}.")
                try:
                    del wellness_managers[session_id]
                except Exception:
                    pass
                return
            await respond_fn("Sorry, I didn't catch that. Could you repeat or say 'save' to save this check-in?")
        except Exception as e:
            logger.exception("Error in handle_wellness: %s", e)
            try:
                await respond_fn("Something went wrong with the wellness flow.")
            except Exception:
                pass

    # ----------------- Tutor handler (Day 4) -----------------
    async def handle_tutor(session_id: str, user_text: str, respond_fn):
        """
        Handles three modes: learn, quiz, teach_back. User can say:
        - 'learn variables' or 'learn loops'
        - 'quiz variables'
        - 'teach back variables'
        - 'list concepts'
        - 'switch to quiz'
        Additionally supports:
        - 'which concepts am i weakest' / 'show weakest'
        - 'show my mastery' / 'my mastery'
        """
        try:
            sess = tutor.start_session(session_id)
            lower = (user_text or "").lower().strip()

            # -------------------------
            # quick mastery queries
            # -------------------------
            if "which concepts am i weakest" in lower or "which concepts am i weakest at" in lower or "weakest concepts" in lower or "show weakest" in lower:
                weakest = tutor.get_weakest(session_id, top_n=3)
                if not weakest:
                    await respond_fn("I don't have any mastery data yet. Try a teach-back or quiz first.")
                    return
                parts = []
                for cid, score in weakest:
                    c = tutor.get_concept(cid)
                    title = c.get("title") if c else cid
                    parts.append(f"{title}: {score}")
                await respond_fn("Your weakest concepts are: " + ", ".join(parts))
                return

            if "show my mastery" in lower or "my mastery" in lower or "show mastery" in lower:
                m = tutor.get_mastery(session_id)
                if not m:
                    await respond_fn("No mastery data yet. Do a quiz or teach-back to start tracking.")
                    return
                lines = []
                for cid, entry in m.items():
                    c = tutor.get_concept(cid)
                    title = (c.get("title") if c else cid)
                    lines.append(f"{title} — avg_score: {entry.get('avg_score')}, last_score: {entry.get('last_score')}, taught_back: {entry.get('times_taught_back')}, quizzed: {entry.get('times_quizzed')}")
                await respond_fn("Here is your mastery: " + " | ".join(lines))
                return

            # mode switching requests: look for "learn", "quiz", "teach back"
            if any(word in lower for word in ("list concepts", "what concepts", "show concepts")):
                concepts = tutor.list_concepts()
                if not concepts:
                    await respond_fn("I don't have any concepts loaded.")
                    return
                out = "I can teach these concepts: " + ", ".join([f'{c["title"]} (id: {c["id"]})' for c in concepts])
                await respond_fn(out)
                return

            # explicit mode + concept: "learn variables", "quiz loops", "teach back variables"
            if lower.startswith("learn ") or lower.startswith("quiz ") or lower.startswith("teach back ") or lower.startswith("teach_back "):
                parts = lower.split()
                mode = parts[0] if parts[0] != "teach" else (parts[0] + " " + parts[1])  # fallback
                if parts[0] == "teach":
                    mode = "teach_back"
                    concept_keyword = " ".join(parts[2:]) if len(parts) > 2 else None
                else:
                    mode = parts[0]
                    concept_keyword = " ".join(parts[1:]) if len(parts) > 1 else None

                # normalize mode
                if mode in ("teach", "teach_back", "teachback", "teach-back"):
                    mode = "teach_back"
                if mode not in ("learn", "quiz", "teach_back"):
                    await respond_fn("I didn't understand that mode. Say learn, quiz, or teach back.")
                    return

                tutor.set_mode(session_id, mode, concept_keyword)
                c = tutor.get_concept(tutor.get_session(session_id)["current_concept"])
                if not c:
                    await respond_fn("I couldn't find that concept.")
                    return

                if mode == "learn":
                    # use per-message voice override for learn (Matthew)
                    # record that we explained this concept
                    tutor.record_explain(session_id, c["id"])
                    await respond_fn(f"Learn mode — {c['title']}: {c['summary']}", voice="en-US-matthew")
                    return

                if mode == "quiz":
                    # quiz voice (Alicia)
                    q = tutor.ask_quiz_question(session_id)
                    await respond_fn(f"Quiz mode — question: {q}", voice="en-US-alicia")
                    return

                if mode == "teach_back":
                    # teach-back voice (Ken)
                    prompt = tutor.ask_teach_back_prompt(session_id)
                    await respond_fn(prompt, voice="en-US-ken")
                    return

            # switch requests "switch to quiz" / "switch to learn"
            if "switch to" in lower or lower.startswith("switch "):
                if "quiz" in lower:
                    tutor.set_mode(session_id, "quiz")
                    q = tutor.ask_quiz_question(session_id)
                    await respond_fn(f"Switched to quiz. {q}", voice="en-US-alicia")
                    return
                if "learn" in lower:
                    tutor.set_mode(session_id, "learn")
                    c = tutor.get_concept(tutor.get_session(session_id)["current_concept"])
                    # record explain exposure
                    if c:
                        tutor.record_explain(session_id, c["id"])
                        await respond_fn(f"Switched to learn. {c['title']}: {c['summary']}", voice="en-US-matthew")
                    else:
                        await respond_fn("Switched to learn but I couldn't find the concept.", voice="en-US-matthew")
                    return
                if "teach" in lower:
                    tutor.set_mode(session_id, "teach_back")
                    p = tutor.ask_teach_back_prompt(session_id)
                    await respond_fn(f"Switched to teach-back. {p}", voice="en-US-ken")
                    return

            # If user answered a quiz question (we assume last_question present)
            sess_state = tutor.get_session(session_id)
            last_q = sess_state.get("last_question") if sess_state else None
            current_cid = sess_state.get("current_concept") if sess_state else None
            if sess_state and sess_state.get("mode") == "quiz" and last_q:
                # evaluate answer via teach_back evaluator
                c = tutor.get_concept(current_cid)
                summary = c.get("summary", "") if c else ""
                eval_res = tutor.evaluate_teach_back(summary, user_text)
                correct = eval_res["score"] >= 60
                tutor.record_quiz_result(session_id, current_cid, correct)
                if correct:
                    await respond_fn(f"Good answer — you included the key ideas. {eval_res['feedback']}", voice="en-US-alicia")
                else:
                    await respond_fn(f"Not quite. {eval_res['feedback']} Here's a quick hint: {summary}", voice="en-US-alicia")
                return

            # If in teach_back mode expecting explanation
            if sess_state and sess_state.get("mode") == "teach_back" and sess_state.get("last_question"):
                c = tutor.get_concept(sess_state.get("current_concept"))
                summary = c.get("summary", "") if c else ""
                eval_res = tutor.evaluate_teach_back(summary, user_text)
                # record taught back score
                if c:
                    tutor.record_taught_back(session_id, c["id"], eval_res["score"])
                await respond_fn(f"I scored your explanation {eval_res['score']} out of 100. {eval_res['feedback']}", voice="en-US-ken")
                return

            # If none of above: see if user asked to start tutoring without explicit mode
            if any(k in lower for k in ("tutor", "teach", "teach me", "i want to learn", "quiz me")):
                # default to an interactive prompt asking which mode
                await respond_fn("Sure — would you like to 'learn' the concept, 'quiz' yourself, or 'teach back'? Say: learn variables, quiz loops, or teach back variables.")
                return

            # fallback: didn't recognize as tutor input
            await respond_fn("Tutor: I didn't quite catch a tutor command. Say 'list concepts' or 'learn variables' or 'quiz loops' or 'teach back variables'.")
        except Exception as e:
            logger.exception("Error in handle_tutor: %s", e)
            try:
                await respond_fn("Something went wrong in the tutor flow.")
            except Exception:
                pass

    # ----------------- SDR handler (Day 5) -----------------
    async def handle_sdr(session_id: str, user_text: str, respond_fn):
        """
        SDR flow:
         - greet
         - answer FAQ driven product questions from shared-data/company_faq_*.json
         - collect lead fields: name, company, email, role, use_case, team_size, timeline
         - on 'save' or 'thanks' -> save lead to backend/leads/ and recite summary
        """
        try:
            # session-local storage (attached to function attr)
            if not hasattr(handle_sdr, "sessions"):
                handle_sdr.sessions = {}
            s = handle_sdr.sessions
            if session_id not in s:
                s[session_id] = new_lead_template()
            lead = s[session_id]
            text = (user_text or "").strip()
            lower = text.lower()

            # new visitor greeting when the session starts and no name known
            if not lead.get("name") and any(k in lower for k in ("hello", "hi", "hey", "start sdr", "i'm interested", "i am interested", "interested")):
                await respond_fn("Hi — thanks for reaching out! I'm the SDR for this demo. What brought you here today? What are you trying to solve?", voice="en-US-matthew")
                return

            # FAQ lookup: attempt best match using shared-data FAQ (robust)
            if any(k in lower for k in ("what does", "product", "pricing", "free", "who is it for", "demo", "what makes", "price", "cost", "pricing", "free trial")):
                try:
                    # look for any company_faq files in shared-data (pick first matching)
                    repo_shared = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared-data"))
                    # prefer exact Freshworks sample if present else first company_faq
                    faq_candidates = [
                        os.path.join(repo_shared, "company_faq_freshworks.json"),
                        os.path.join(repo_shared, "company_faq_sample.json"),
                        os.path.join(repo_shared, "company_faq.json"),
                    ]
                    faqs = None
                    for p in faq_candidates:
                        if os.path.exists(p):
                            with open(p, "r", encoding="utf-8") as f:
                                faqs = json.load(f)
                            break
                    # fallback: any json file in shared-data
                    if faqs is None:
                        for fname in os.listdir(repo_shared) if os.path.isdir(repo_shared) else []:
                            pass
                    if not faqs:
                        await respond_fn("I don't have product info available right now. Would you like me to schedule a demo?", voice="en-US-matthew")
                        return

                    best, score = best_faq_for_text(lower, faqs)
                    if best and score > 0:
                        # return exact FAQ answer (short)
                        await respond_fn(best.get("answer", "Sorry I don't have that detail."), voice="en-US-matthew")
                        return
                    # fallback: keyword-based small-match
                    for fentry in faqs:
                        q = (fentry.get("question") or "").lower()
                        tokens = [t for t in re.split(r"\W+", q) if t]
                        if any(tok in lower for tok in tokens[:4]):
                            await respond_fn(fentry.get("answer", ""), voice="en-US-matthew")
                            return
                    # final fallback
                    await respond_fn("I can answer product/pricing questions or schedule a demo. What would you like to know specifically?", voice="en-US-matthew")
                except Exception as exc:
                    logger.exception("FAQ lookup error: %s", exc)
                    await respond_fn("Sorry — I couldn't load product info right now.", voice="en-US-matthew")
                return

            # collect name
            if (not lead.get("name")) and re.search(r"\b(my name is|i am|i'm)\b", lower):
                try:
                    # capture after phrase
                    m = re.search(r"\b(?:my name is|i am|i'm)\b\s*(?P<name>[A-Za-z\-']+)", text, re.IGNORECASE)
                    if m:
                        name = m.group("name").strip().capitalize()
                    else:
                        name = text.split()[:1][0].capitalize()
                    if name:
                        lead["name"] = name
                        await respond_fn(f"Nice to meet you, {name}. Which company are you with?", voice="en-US-matthew")
                        return
                except Exception:
                    pass

            # collect company by common patterns: "from X", "we are X", "at X", "company is X"
            if (not lead.get("company")) and any(k in lower for k in ("company", "we are", "from", "i work at", "at ")):
                m = re.search(r"(?:from|at|company is|we are|we're)\s+([A-Za-z0-9\-\&\s\.]{2,40})", text, re.IGNORECASE)
                if m:
                    candidate = m.group(1).strip().strip(".")
                    company = " ".join(candidate.split()[:6])
                    lead["company"] = company
                    await respond_fn(f"Got it — {company}. What's your role there?", voice="en-US-matthew")
                    return
                await respond_fn("Which company are you with?", voice="en-US-matthew")
                return

            # email capture (simple)
            if (not lead.get("email")) and re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text):
                m = re.search(r"([\w\.-]+@[\w\.-]+\.\w+)", text)
                lead["email"] = m.group(1)
                await respond_fn("Thanks — noted your email. What is your role / title?", voice="en-US-matthew")
                return

            # role capture (common keywords)
            if (not lead.get("role")) and any(k in lower for k in ("engineer", "developer", "product", "founder", "ceo", "cto", "manager", "pm", "head", "director")):
                lead["role"] = text.strip()
                await respond_fn("Great — could you briefly describe what you'd like to use the product for? (1-2 sentences)", voice="en-US-matthew")
                return

            # use_case capture (assume a longer sentence)
            if (not lead.get("use_case")):
                if len(text.split()) > 6:
                    lead["use_case"] = text.strip()
                    await respond_fn("Thanks — that helps. How big is your team? (e.g. 1-10, 10-50, 50+)", voice="en-US-matthew")
                    return

            # team size
            if (not lead.get("team_size")) and any(k in lower for k in ("team", "people", "employees", "1-10", "10-50", "50+")):
                lead["team_size"] = text.strip()
                await respond_fn("When would you like to start this — now, soon (1-3 months), or later?", voice="en-US-matthew")
                return

            # timeline
            if (not lead.get("timeline")) and any(k in lower for k in ("now", "soon", "later", "1-3", "month", "months")):
                lead["timeline"] = text.strip()
                await respond_fn("Great — thanks. Anything else you'd like to add or should I save this lead and summary?", voice="en-US-matthew")
                return

            # explicit save / end of call
            if any(k in lower for k in ("save", "that is all", "i'm done", "i am done", "thanks", "thank you", "that's all")):
                lead["collected_at"] = datetime.now().isoformat()
                # ensure some fields exist for readability
                if not lead.get("company"):
                    lead["company"] = "Unknown"
                path = save_lead(lead)
                summary = build_summary(lead)
                await respond_fn(f"Thanks — I saved the lead. Summary: {summary} I stored it at {path}. Our SDR team will follow up at {lead.get('email','the provided email')}.", voice="en-US-matthew")
                # clear session
                try:
                    del handle_sdr.sessions[session_id]
                except Exception:
                    pass
                return

            # guided prompts for missing fields
            if not lead.get("name"):
                await respond_fn("Before we go further — what's your name?", voice="en-US-matthew")
                return
            if not lead.get("company"):
                await respond_fn("Which company do you work for?", voice="en-US-matthew")
                return
            if not lead.get("email"):
                await respond_fn("What's the best email to reach you at?", voice="en-US-matthew")
                return
            if not lead.get("use_case"):
                await respond_fn("Can you tell me briefly what you'd like to use this for?", voice="en-US-matthew")
                return

            await respond_fn("Thanks — I didn't catch that. Could you rephrase or say 'save' to finish and store your details?", voice="en-US-matthew")
        except Exception as e:
            logger.exception("SDR handler error: %s", e)
            try:
                await respond_fn("Sorry, something went wrong in the SDR flow.", voice="en-US-matthew")
            except Exception:
                pass

    # ----------------- Unified incoming handler (routes to flows) -----------------
    async def _handle_incoming_event(ev):
        try:
            logger.info(">>>> INCOMING EVENT FIRED <<<<")
            logger.debug("RAW EVENT: %r", ev)
            text = None
            session_id = None
            if hasattr(ev, "text"):
                text = ev.text
            elif hasattr(ev, "transcript"):
                text = ev.transcript
            elif hasattr(ev, "alternatives") and ev.alternatives:
                alt0 = ev.alternatives[0]
                text = getattr(alt0, "transcript", None) or getattr(alt0, "text", None)
            elif isinstance(ev, dict):
                for key in ("text", "transcript", "message", "body"):
                    if key in ev:
                        candidate = ev[key]
                        if isinstance(candidate, dict):
                            text = candidate.get("text") or candidate.get("transcript")
                        else:
                            text = candidate
                        if text:
                            break
                if not text and "alternatives" in ev and ev["alternatives"]:
                    alt0 = ev["alternatives"][0]
                    if isinstance(alt0, dict):
                        text = alt0.get("transcript") or alt0.get("text")
            if not text:
                try:
                    msg = getattr(ev, "message", None)
                    if msg:
                        text = getattr(msg, "text", None) or (msg.get("text") if isinstance(msg, dict) else None)
                except Exception:
                    pass
            try:
                if hasattr(ev, "participant") and ev.participant is not None:
                    session_id = getattr(ev.participant, "identity", None) or getattr(ev.participant, "sid", None)
            except Exception:
                session_id = None
            if not session_id:
                try:
                    session_id = ctx.room.name
                except Exception:
                    session_id = "default"
            logger.info("EXTRACTED text: %s", repr(text))
            logger.info("SESSION_ID used: %s", session_id)
            if not text:
                logger.info("No text found in incoming event - ignoring.")
                return
            lower = text.lower() if isinstance(text, str) else ""

            # routing priority:
            # SDR -> tutor -> wellness -> coffee
            sdr_triggers = ("sales", "pricing", "demo", "book demo", "interested", "contact", "sdr", "lead", "price", "cost", "product", "trial", "free")
            tutor_triggers = ("tutor", "teach me", "teach back", "quiz me", "learn", "quiz", "teach back")
            wellness_triggers = ("check in", "wellness", "daily check", "start wellness", "how are you feeling", "how am i")

            # prefer SDR if product/pricing/demo keywords present
            if any(kw in lower for kw in sdr_triggers):
                logger.info("Routing to SDR flow")
                await handle_sdr(session_id, text, respond_fn_factory(session, ctx))
            elif any(kw in lower for kw in tutor_triggers):
                logger.info("Routing to tutor flow")
                await handle_tutor(session_id, text, respond_fn_factory(session, ctx))
            elif any(kw in lower for kw in wellness_triggers):
                logger.info("Routing to wellness flow")
                await handle_wellness(session_id, text, respond_fn_factory(session, ctx))
            else:
                logger.info("Routing to coffee flow")
                await handle_coffee(session_id, text, respond_fn_factory(session, ctx))
        except Exception as e:
            logger.exception("Exception in unified handler: %s", e)

    # register handlers
    try:
        @session.on("transcript")
        async def _on_transcript(ev):
            await _handle_incoming_event(ev)
    except Exception:
        logger.debug("Failed to attach handler for 'transcript'")
    try:
        @session.on("transcription")
        async def _on_transcription(ev):
            await _handle_incoming_event(ev)
    except Exception:
        logger.debug("Failed to attach handler for 'transcription'")
    try:
        @session.on("message")
        async def _on_message(ev):
            await _handle_incoming_event(ev)
    except Exception:
        logger.debug("Failed to attach handler for 'message'")

    # mention last wellness entry non-intrusively
    try:
        last = WellnessManager.last_entry()
        if last:
            short = last.get("summary") or f"mood: {last.get('mood')}, energy: {last.get('energy')}"
            try:
                if hasattr(session, "send_text"):
                    await session.send_text(f"Welcome back — last time you said: {short}. Would you like to do today's check-in?")
                else:
                    await ctx.room.send_data(f"Welcome back — last time you said: {short}. Would you like to do today's check-in?")
            except Exception:
                logger.debug("Could not notify room about previous check-in")
    except Exception:
        logger.debug("No previous wellness entry or error reading it.")

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
