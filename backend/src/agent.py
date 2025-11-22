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
You are a friendly barista when the user places an order: ask clarifying questions until the order is complete.
Keep responses concise and friendly.""",
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
    # We keep a dict mapping room/session identifier -> OrderManager
    order_managers: dict[str, OrderManager] = {}

    def get_mgr(session_id: str) -> OrderManager:
        if session_id not in order_managers:
            order_managers[session_id] = OrderManager()
        return order_managers[session_id]

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

    # ----------------- Transcription event handler -----------------
    # NOTE: event name 'transcript' is a best-effort choice compatible with typical
    # livekit-agents versions. If your installed version uses a different event name,
    # replace "transcript" with the correct event name (for example "transcription").
    @session.on("transcript")
    async def _on_transcript(ev):
        """
        Called when a new transcript chunk/turn arrives.
        We attempt to extract text and participant identity, and then drive the order flow.
        """
        try:
            # Several SDK versions use different attribute names; be defensive.
            text = None
            session_id = None

            # event may expose multiple attributes; try common ones
            if hasattr(ev, "text"):
                text = ev.text
            elif hasattr(ev, "transcript"):
                text = ev.transcript
            elif isinstance(ev, dict) and "text" in ev:
                text = ev["text"]

            # determine session id / participant identity
            if hasattr(ev, "participant") and ev.participant is not None:
                # some participant objects expose `identity` or `sid`
                session_id = getattr(ev.participant, "identity", None) or getattr(ev.participant, "sid", None)
            # fallback to room name
            if not session_id:
                try:
                    session_id = ctx.room.name
                except Exception:
                    session_id = "default"

            if not text:
                # nothing to process
                return

            # define respond function: try several possible send methods supported by session
            async def respond_fn(reply_text: str):
                # try a few ways to send a textual reply that triggers TTS
                sent = False
                try:
                    # Preferred: session.send_text (if present)
                    if hasattr(session, "send_text") and callable(session.send_text):
                        await session.send_text(reply_text)
                        sent = True
                except Exception:
                    logger.debug("session.send_text not available or failed")

                if not sent:
                    try:
                        # Some SDKs expose `publish_text` or `send` APIs
                        if hasattr(session, "publish_text") and callable(session.publish_text):
                            await session.publish_text(reply_text)
                            sent = True
                    except Exception:
                        logger.debug("session.publish_text not available or failed")

                if not sent:
                    try:
                        # Try agent-level send if available
                        agent_obj = getattr(session, "agent", None)
                        if agent_obj and hasattr(agent_obj, "send_message"):
                            await agent_obj.send_message(reply_text)
                            sent = True
                    except Exception:
                        logger.debug("agent.send_message not available or failed")

                if not sent:
                    # Last resort: publish to the room as a data message (room-level)
                    try:
                        await ctx.room.send_data(reply_text)
                        sent = True
                    except Exception:
                        logger.exception("Failed to send reply via any available method")

            # Finally call the coffee handler
            await handle_coffee(session_id, text, respond_fn)

        except Exception as e:
            logger.exception("transcript handler error: %s", e)

    # connect to LiveKit room and start
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
