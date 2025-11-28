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

# lead manager for SDR (Day 5) - keep if present
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
# ----------------------- Food Cart + Catalog Manager (Day 7) -----------------------
#
class CatalogManager:
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self.catalog = []
        self.by_id = {}
        self.load_catalog()

    def load_catalog(self):
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)
            self.by_id = {item["id"]: item for item in self.catalog}
        except Exception:
            logger.exception("Failed to load catalog at %s", self.catalog_path)
            self.catalog = []
            self.by_id = {}

    def find_by_name(self, name: str):
        name = (name or "").lower()
        # exact match id/name/token match
        for item in self.catalog:
            if name == item.get("id", "").lower() or name == item.get("name", "").lower():
                return item
        # token match
        tokens = [t for t in re.split(r"\W+", name) if t]
        for item in self.catalog:
            item_text = (item.get("name", "") + " " + " ".join(item.get("tags", []))).lower()
            if any(tok in item_text for tok in tokens):
                return item
        return None

    def lookup(self, item_id: str):
        return self.by_id.get(item_id)


class CartManager:
    def __init__(self):
        # cart: dict item_id -> {item, qty}
        self.cart = {}

    def add_item(self, item, qty=1):
        if not item or qty <= 0:
            return
        iid = item["id"]
        if iid in self.cart:
            self.cart[iid]["qty"] += int(qty)
        else:
            self.cart[iid] = {"item": item, "qty": int(qty)}

    def remove_item(self, item_id):
        if item_id in self.cart:
            del self.cart[item_id]

    def update_qty(self, item_id, qty):
        if item_id in self.cart:
            if qty <= 0:
                del self.cart[item_id]
            else:
                self.cart[item_id]["qty"] = int(qty)

    def list_items(self):
        return [{"id": v["item"]["id"], "name": v["item"]["name"], "qty": v["qty"], "price": v["item"].get("price", 0)} for v in self.cart.values()]

    def total(self):
        return sum(v["qty"] * v["item"].get("price", 0) for v in self.cart.values())

    def is_empty(self):
        return len(self.cart) == 0

    def clear(self):
        self.cart = {}

    def to_order_object(self, customer_name=None, address=None):
        items = self.list_items()
        return {
            "order_id": f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "customer_name": customer_name or "guest",
            "address": address or "",
            "items": items,
            "total": self.total(),
            "timestamp": datetime.now().isoformat(),
        }

    def save_order(self, folder="orders", customer_name=None, address=None):
        order = self.to_order_object(customer_name=customer_name, address=address)
        os.makedirs(folder, exist_ok=True)
        filename = f"{order['order_id']}.json"
        path = os.path.join(folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(order, f, indent=2, ensure_ascii=False)
        return path, order


#
# ----------------------- Agent / Entry point -----------------------
#
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice.
You can act as a friendly barista, a grounded wellness companion, an active recall tutor, or a food ordering assistant.
Keep responses concise and practical.""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# small recipes mapping for "ingredients for X" requests
RECIPES = {
    "peanut butter sandwich": ["bread_wholewheat", "peanut_butter_jar"],
    "pasta for two": ["pasta_500g", "pasta_sauce", "butter_200g"],
    "sandwich": ["bread_wholewheat", "sandwich_cheese", "tomato_ketchup"]
}


# helper: robust FAQ matching (reused from earlier flows)
def best_faq_for_text(user_text: str, faq_list: list[dict]) -> tuple[dict | None, int]:
    if not user_text or not faq_list:
        return None, 0
    t = user_text.lower()
    best = None
    best_score = 0
    for f in faq_list:
        q = (f.get("question") or "").lower()
        a = (f.get("answer") or "").lower()
        tokens = [tok for tok in re.split(r"\W+", q) if tok]
        tokens += [tok for tok in re.split(r"\W+", a) if tok][:6]
        score = sum(1 for tok in set(tokens) if tok and tok in t)
        if score > best_score:
            best_score = score
            best = f
    return best, best_score


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # set paths
    repo_shared = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared-data"))
    catalog_path = os.path.join(repo_shared, "catalog_zepto.json")
    catalog_mgr = CatalogManager(catalog_path)

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
    tutor_sessions = {}
    # food carts per session
    food_carts: dict[str, CartManager] = {}

    def get_mgr(session_id: str) -> OrderManager:
        if session_id not in order_managers:
            order_managers[session_id] = OrderManager()
        return order_managers[session_id]

    def get_wellness_mgr(session_id: str) -> WellnessManager:
        if session_id not in wellness_managers:
            wellness_managers[session_id] = WellnessManager()
        return wellness_managers[session_id]

    def get_cart(session_id: str) -> CartManager:
        if session_id not in food_carts:
            food_carts[session_id] = CartManager()
        return food_carts[session_id]

    # respond factory
    def respond_fn_factory(sess, ctx_obj):
        async def respond_fn(reply_text: str, voice: str | None = None):
            sent = False
            if voice:
                payload = {"text": reply_text, "tts": {"voice": voice}}
                try:
                    if hasattr(sess, "send_text") and callable(sess.send_text):
                        await sess.send_text(payload)
                        return
                except Exception:
                    pass
                try:
                    if hasattr(sess, "publish_text") and callable(sess.publish_text):
                        await sess.publish_text(payload)
                        return
                except Exception:
                    pass
            try:
                if hasattr(sess, "send_text") and callable(sess.send_text):
                    await sess.send_text(reply_text)
                    sent = True
            except Exception:
                pass
            if not sent:
                try:
                    if hasattr(sess, "publish_text") and callable(sess.publish_text):
                        await sess.publish_text(reply_text)
                        sent = True
                except Exception:
                    pass
            if not sent:
                try:
                    agent_obj = getattr(sess, "agent", None)
                    if agent_obj and hasattr(agent_obj, "send_message"):
                        await agent_obj.send_message(reply_text)
                        sent = True
                except Exception:
                    pass
            if not sent:
                try:
                    await ctx_obj.room.send_data(reply_text)
                    sent = True
                except Exception:
                    logger.exception("Failed to send reply via any available method")
        return respond_fn

    # ----------------- Food ordering handler (Day 7) -----------------
    async def handle_food(session_id: str, user_text: str, respond_fn):
        """
        Food ordering flow:
        - interpret add/remove/list/place commands
        - support 'ingredients for X'
        - save order JSON when user says 'place order' or 'that's all'
        """
        try:
            cart = get_cart(session_id)
            text = (user_text or "").strip()
            lower = text.lower()

            # list catalog sample
            if any(k in lower for k in ("what do you have", "show catalog", "menu", "what can i order")):
                # list top categories/items (short)
                sample = [f"{i['name']} — ₹{i.get('price', '?')}" for i in catalog_mgr.catalog[:8]]
                await respond_fn("I have: " + "; ".join(sample) + ". You can say 'add 2 bread' or 'ingredients for peanut butter sandwich'.")
                return

            # show cart
            if any(k in lower for k in ("what's in my cart", "show cart", "what is in my cart", "view cart", "cart")):
                if cart.is_empty():
                    await respond_fn("Your cart is empty. Want me to add something? Try 'add peanut butter'.")
                    return
                items = cart.list_items()
                lines = [f"{it['qty']} x {it['name']} (₹{it['price']} each)" for it in items]
                await respond_fn("In your cart: " + " ; ".join(lines) + f". Total: ₹{cart.total()}. Say 'place order' to finalize.")
                return

            # place order
            if any(k in lower for k in ("place order", "i'm done", "i am done", "that's all", "checkout", "order now")):
                if cart.is_empty():
                    await respond_fn("Your cart is empty — nothing to place. Say 'add bread' to add something first.")
                    return
                # optional: try to extract a name/address from session text or ask
                name_match = re.search(r"\b(name is|i am|i'm)\s+([A-Za-z\- ]{2,40})", text, re.IGNORECASE)
                name = None
                if name_match:
                    name = name_match.group(2).strip()
                # save and clear
                path, order_obj = cart.save_order(folder=os.path.join(os.path.dirname(__file__), "..", "orders"), customer_name=name)
                await respond_fn(f"Order placed. Summary: {len(order_obj['items'])} items, total ₹{order_obj['total']}. Saved to {path}")
                cart.clear()
                return

            # remove item: "remove bread" or "remove 1 bread"
            if lower.startswith("remove") or lower.startswith("delete") or lower.startswith("remove "):
                # get name
                m = re.search(r"(?:remove|delete)\s+(\d+)\s+(.+)", text, re.IGNORECASE)
                if m:
                    qty = int(m.group(1))
                    target = m.group(2).strip()
                else:
                    # remove all of named item
                    m2 = re.search(r"(?:remove|delete)\s+(.+)", text, re.IGNORECASE)
                    qty = None
                    target = m2.group(1).strip() if m2 else None
                if target:
                    target_item = catalog_mgr.find_by_name(target)
                    if target_item:
                        iid = target_item["id"]
                        if qty is None:
                            cart.remove_item(iid)
                            await respond_fn(f"Removed {target_item['name']} from your cart.")
                        else:
                            # reduce quantity
                            existing = cart.cart.get(iid)
                            if existing:
                                newq = max(0, existing["qty"] - qty)
                                cart.update_qty(iid, newq)
                                await respond_fn(f"Updated {target_item['name']} quantity to {newq}.")
                            else:
                                await respond_fn(f"I couldn't find {target_item['name']} in your cart.")
                        return
                    else:
                        await respond_fn("I couldn't find that item in the catalog.")
                        return

            # add by 'ingredients for X'
            if any(k in lower for k in ("ingredients for", "ingredients to make", "ingredients for a", "ingredients for an")):
                # extract dish name
                m = re.search(r"ingredients (?:for|to make)\s+(.+)", lower)
                dish = m.group(1).strip() if m else lower.replace("ingredients for", "").strip()
                # try direct recipe match
                recipe_key = dish
                if recipe_key in RECIPES:
                    for iid in RECIPES[recipe_key]:
                        item = catalog_mgr.lookup(iid)
                        if item:
                            cart.add_item(item, qty=1)
                    await respond_fn(f"Added ingredients for {dish}: " + ", ".join([catalog_mgr.lookup(i)["name"] for i in RECIPES[recipe_key] if catalog_mgr.lookup(i)] ) + ".")
                    return
                # try fuzzy: match dish tokens against recipe keys
                for key in RECIPES.keys():
                    if all(tok in key for tok in re.split(r"\W+", dish) if tok):
                        for iid in RECIPES[key]:
                            item = catalog_mgr.lookup(iid)
                            if item:
                                cart.add_item(item, qty=1)
                        await respond_fn(f"Added ingredients for {key}.")
                        return
                await respond_fn("I don't have a recipe mapping for that dish, but I can add items if you say them by name.")
                return

            # add item phrases: "add 2 bread", "i want 3 pasta", "add peanut butter"
            m_add = re.search(r"(?:add|put|i want|i'd like|i want to add)\s*(\d+)?\s*(.+)", text, re.IGNORECASE)
            if m_add:
                qty = int(m_add.group(1)) if m_add.group(1) else 1
                target = m_add.group(2).strip()
                # clean punctuation
                target = re.sub(r"[\.!?]$", "", target).strip()
                # try find item
                item = catalog_mgr.find_by_name(target)
                if item:
                    cart.add_item(item, qty=qty)
                    await respond_fn(f"Added {qty} x {item['name']} to your cart. Current total ₹{cart.total()}.")
                    return
                # If user gave an item id maybe
                item_by_id = catalog_mgr.lookup(target)
                if item_by_id:
                    cart.add_item(item_by_id, qty=qty)
                    await respond_fn(f"Added {qty} x {item_by_id['name']} to your cart. Current total ₹{cart.total()}.")
                    return
                await respond_fn("I couldn't find that item in the catalog. Try another name (e.g., 'add peanut butter').")
                return

            # quick update qty pattern: "change bread to 2"
            m_upd = re.search(r"(?:update|change|set)\s+(.+?)\s+(?:to|=)\s*(\d+)", text, re.IGNORECASE)
            if m_upd:
                name = m_upd.group(1).strip()
                qty = int(m_upd.group(2))
                it = catalog_mgr.find_by_name(name)
                if it:
                    cart.update_qty(it["id"], qty)
                    await respond_fn(f"Updated {it['name']} quantity to {qty}. Total ₹{cart.total()}.")
                    return
                await respond_fn("Couldn't find that item to update.")
                return

            # If nothing matched but user said 'food' or 'order' start ordering prompt
            if any(k in lower for k in ("food", "grocery", "groceries", "order food", "order groceries", "i want groceries", "i want food")):
                await respond_fn("I can add items to your cart. Say: 'add peanut butter', 'ingredients for peanut butter sandwich', 'show cart', or 'place order'.")
                return

            # fallback fallback for food route: ask clarifying
            await respond_fn("For ordering: say 'add <item>' or 'ingredients for <dish>' or 'show cart' or 'place order'.")
        except Exception as e:
            logger.exception("Error in handle_food: %s", e)
            try:
                await respond_fn("Sorry, something went wrong with the ordering flow.")
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
            # food -> SDR -> tutor -> wellness -> coffee
            food_triggers = ("add", "ingredients", "place order", "show cart", "menu", "catalog", "order", "grocery", "groceries", "food")
            sdr_triggers = ("sales", "pricing", "demo", "book demo", "interested", "contact", "sdr", "lead", "price", "cost", "product", "trial", "free")
            tutor_triggers = ("tutor", "teach me", "teach back", "quiz me", "learn", "quiz", "teach back")
            wellness_triggers = ("check in", "wellness", "daily check", "start wellness", "how are you feeling", "how am i")

            if any(kw in lower for kw in food_triggers):
                logger.info("Routing to food flow")
                await handle_food(session_id, text, respond_fn_factory(session, ctx))
            elif any(kw in lower for kw in sdr_triggers):
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
