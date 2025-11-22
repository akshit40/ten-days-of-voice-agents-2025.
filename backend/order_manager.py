import json
import os
from datetime import datetime

class Manager:
    def __init__(self):
        self. = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": ""
        }

    def update(self, text):
        t = text.lower()

        # drink type
        for d in ["latte", "cappuccino", "americano", "espresso", "mocha"]:
            if d in t:
                self.["drinkType"] = d

        # size
        for s in ["small", "medium", "large"]:
            if s in t:
                self.["size"] = s

        # milk
        for m in ["whole", "skim", "oat", "soy", "almond"]:
            if m in t:
                self.["milk"] = m

        # extras
        for e in ["vanilla", "caramel", "hazelnut", "whipped"]:
            if e in t and e not in self.["extras"]:
                self.["extras"].append(e)

        # name
        if "my name is" in t:
            self.["name"] = t.split("my name is")[-1].strip().split()[0]
        if "for" in t:
            self.["name"] = t.split("for")[-1].strip().split()[0]

    def is_complete(self):
        return (
            self.["drinkType"]
            and self.["size"]
            and self.["milk"]
            and self.["name"]
        )

    def next_question(self):
        if not self.["drinkType"]:
            return "What drink would you like?"
        if not self.["size"]:
            return "What size do you prefer?"
        if not self.["milk"]:
            return "What milk would you like?"
        if len(self.["extras"]) == 0:
            return "Any extras like caramel or whipped cream?"
        if not self.["name"]:
            return "What name should I put on your ?"
        return None

    def save(self):
        os.makedirs("s", exist_ok=True)
        filename = f"s/_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self., f, indent=2)
        return filename
