try:
    import app
    print("IMPORT OK")
except Exception:
    import traceback
    traceback.print_exc()