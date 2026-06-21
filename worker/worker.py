# worker/worker.py
# This is a tiny helper to show the RQ worker entrypoint is present.
# In our docker-compose we run "rq worker", so this file is optional.
# But keep it for reference and potential custom worker code.

if __name__ == "__main__":
    print("This file is a placeholder. The real worker runs with 'rq worker'.")
