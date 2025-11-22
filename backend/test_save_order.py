from order_manager import OrderManager

mgr = OrderManager()
mgr.order = {
    "drinkType": "latte",
    "size": "large",
    "milk": "oat",
    "extras": ["caramel"],
    "name": "Akshit"
}
path = mgr.save()
print("Saved order to:", path)
