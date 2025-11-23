import logging
import json
import os
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

# Import the WellnessManager (make sure backend/src/wellness_manager.py exists)
from wellness_manager import WellnessManager

logger = logging.getLogger("agent")
load_dotenv(".env.local")


#
# ----------------------- OrderManager -----------------------
#
class OrderManager:
    """
    Very small slot-filling order manager. Keeps the simple order state,
    extracts naive keywords from text, asks the next question, and can save to JSON.
    """

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

        # Drink types
        for d in ["latte", "cappuccino", "americano", "espresso", "mocha", "cold brew", "flat white"]:
            if d in t:
                self.order["drinkType"] = d

        # Size
        for s in ["small", "medium", "large"]:
            if s in t:
                self.order["size"] = s

        # Milk
        for m in ["whole", "skim", "oat", "soy", "almond", "2%"]:
            if m in t:
                self.order["milk"] = m

        # Extras (append, avoid duplicates)
        for ex in ["vanilla", "caramel", "hazelnut", "whipped", "extra shot", "shot"]:
            if ex in t and ex not in self.order["extras"]:
                self.order["extras"].append(ex)

        # Name detection (naive patterns)
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
        # extras may be empty list; require drinkType, size, milk, name
        return bool(self.order["drinkType"] and self.order["size"] and self.order["milk"] and self.order["name"])

    def next_question(self) -> str | None:
        if not self.order["drinkType"]:
            return "What would you like to drink today? We have latte, cappuccino, americano, mocha, and espresso."
        if not self.order["size"]:
            return "What size would you like — small, medium, or large?"
        if not self.order["milk"]:
            return "Which milk would you prefer — whole, skim, oat, soy or almond?"
        # extras optional: ask only if none chosen
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
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice.
You can act as a friendly barista for coffee orders, or as a grounded wellness companion when asked.
Keep responses concise, calm, and practical. Avoid medical advice — offer only simple, non-diagnostic support.""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context fields
    ctx.log_context_fields = {"room": ctx.room.name}

    # Build the voice AI pipeline
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

    # Metrics collector
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session and join the room
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # ----------------- Simple session-scoped order managers -----------------
    # maps room/session identifier -> OrderManager
    order_managers: dict[str, OrderManager] = {}

    def get_mgr(session_id: str) -> OrderManager:
        if session_id not in order_managers:
            order_managers[session_id] = OrderManager()
        return order_managers[session_id]

    # ----------------- Wellness managers -----------------
    wellness_managers: dict[str, WellnessManager] = {}

    def get_wellness_mgr(session_id: str) -> WellnessManager:
        if session_id not in wellness_managers:
            wellness_managers[session_id] = WellnessManager()
        return wellness_managers[session_id]

    # ----------------- Respond function factory -----------------
    def respond_fn_factory(sess, ctx_obj):
        async def respond_fn(reply_text: str):
            sent = False
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

    # ----------------- Coffee handler -----------------
    async def handle_coffee(session_id: str, user_text: str, respond_fn):
        """
        Update order manager with user_text, ask next question, save when complete.
        respond_fn should be an async callable that accepts a single text string and
        will result in the agent speaking that text (TTS).
        """
        try:
            mgr = get_mgr(session_id)
            mgr.update_from_text(user_text)

            # If complete after update, save and notify
            if mgr.is_complete():
                path = mgr.save()
                summary_text = (
                    f"Perfect — your order is a {mgr.order['size']} {mgr.order['drinkType']} "
                    f"with {mgr.order['milk']} milk"
                )
                if mgr.order["extras"]:
                    summary_text += f" and extras: {', '.join(mgr.order['extras'])}"
                summary_text += f" for {mgr.order['name']}. I've saved it to {path}. Enjoy!"
                await respond_fn(summary_text)
                # clear manager for new orders
                try:
                    del order_managers[session_id]
                except Exception:
                    pass
                return

            # Otherwise ask next clarifying question
            q = mgr.next_question()
            if q:
                await respond_fn(q)
                return

            # fallback generic prompt
            await respond_fn("Sorry, I didn't catch that. Could you repeat please?")
        except Exception as e:
            logger.exception("Error in handle_coffee: %s", e)
            try:
                await respond_fn("Something went wrong processing your order.")
            except Exception:
                pass

    # ----------------- Wellness handler -----------------
    async def handle_wellness(session_id: str, user_text: str, respond_fn):
        """
        Conduct a short wellness check-in via voice using WellnessManager.
        """
        try:
            mgr = get_wellness_mgr(session_id)

            # Update with user text
            mgr.update_from_text(user_text)

            # Ask next question if any
            q = mgr.next_question()
            if q:
                await respond_fn(q)
                # advance asked index so next reply maps to next field
                mgr._asked_index += 1
                return

            # If ready to confirm and not yet completed, build summary and ask for save
            if mgr.is_ready_to_confirm() and not mgr.is_complete():
                summary = mgr.build_summary()
                await respond_fn(f"Quick summary: {summary} Do you want me to save this check-in?")
                mgr._asked_index = len(mgr.QUESTIONS) - 1
                return

            # If user confirmed (complete)
            if mgr.is_complete():
                path, saved_entry = mgr.save()
                await respond_fn(
                    f"Saved today's check-in. Summary: {saved_entry.get('summary','')}. I saved it to {path}."
                )
                try:
                    del wellness_managers[session_id]
                except Exception:
                    pass
                return

            # fallback
            await respond_fn("Sorry, I didn't catch that. Could you repeat or say 'save' to save this check-in?")
        except Exception as e:
            logger.exception("Error in handle_wellness: %s", e)
            try:
                await respond_fn("Something went wrong with the wellness flow.")
            except Exception:
                pass

    # ----------------- Unified defensive incoming event handler -----------------
    async def _handle_incoming_event(ev):
        """
        Extract text from a variety of event shapes, log the raw event, and route to the correct flow.
        """
        try:
            logger.info(">>>> INCOMING EVENT FIRED <<<<")
            logger.debug("RAW EVENT: %r", ev)

            text = None
            session_id = None

            # common shapes: attributes
            if hasattr(ev, "text"):
                text = ev.text
            elif hasattr(ev, "transcript"):
                text = ev.transcript

            # alternatives (common STT shape)
            elif hasattr(ev, "alternatives") and ev.alternatives:
                alt0 = ev.alternatives[0]
                text = getattr(alt0, "transcript", None) or getattr(alt0, "text", None)

            # dict-like event
            elif isinstance(ev, dict):
                # try several keys
                for key in ("text", "transcript", "message", "body"):
                    if key in ev:
                        candidate = ev[key]
                        if isinstance(candidate, dict):
                            text = candidate.get("text") or candidate.get("transcript")
                        else:
                            text = candidate
                        if text:
                            break
                # alternatives in dict
                if not text and "alternatives" in ev and ev["alternatives"]:
                    alt0 = ev["alternatives"][0]
                    if isinstance(alt0, dict):
                        text = alt0.get("transcript") or alt0.get("text")

            # deeper fallback: ev.message.text
            if not text:
                try:
                    msg = getattr(ev, "message", None)
                    if msg:
                        text = getattr(msg, "text", None) or (msg.get("text") if isinstance(msg, dict) else None)
                except Exception:
                    pass

            # participant/session id detection
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

            # normalize text
            lower = text.lower() if isinstance(text, str) else ""

            # choose flow by keywords
            wellness_keywords = ("check in", "wellness", "daily check", "start wellness", "how are you feeling", "how am i")
            if any(kw in lower for kw in wellness_keywords):
                logger.info("Routing to wellness flow")
                await handle_wellness(session_id, text, respond_fn_factory(session, ctx))
            else:
                # default to coffee flow
                logger.info("Routing to coffee flow")
                await handle_coffee(session_id, text, respond_fn_factory(session, ctx))

        except Exception as e:
            logger.exception("Exception in unified handler: %s", e)

    # ----------------- Register handler for common event names -----------------
    # Attempt to attach to multiple possible event names used by different SDK versions.
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

    # Optional: at session start, briefly reference last wellness entry (non-intrusive)
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

    # connect to LiveKit room and start listening
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
