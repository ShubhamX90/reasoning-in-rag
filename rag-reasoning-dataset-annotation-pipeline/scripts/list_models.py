# quick check: save as scripts/list_models.py and run with your venv
from openai import OpenAI
client = OpenAI()  # uses OPENAI_API_KEY
ids = [m.id for m in client.models.list().data]
for mid in sorted(ids):
    if mid.startswith("gpt-5") or mid.startswith("gpt-4"):
        print(mid)