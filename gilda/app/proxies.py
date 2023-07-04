from werkzeug.local import LocalProxy

from flask import current_app

__all__ = [
    "grounder",
]

# The way that local proxies work is that when the app gets
# instantiated, you can stick objects into the `app.config`
# dictionary, then the local proxy lets you access them through
# a fake "current_app" object.
grounder = LocalProxy(lambda: current_app.config["grounder"])
