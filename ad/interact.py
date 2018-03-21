import importlib

from ad import conn, gen, model, vis


def reload():
    print("Reloading modules...")
    importlib.reload(conn)
    importlib.reload(gen)
    importlib.reload(model)
    importlib.reload(models)
    importlib.reload(vis)
