import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = "http://localhost:8000/v1"
client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)

response = client.chat.completions.create(
    model="rstar-gpt-4o-mini",  # Base model name
    messages=[
        {
            "role": "user",
            "content": "For how many two-digit primes is the sum of the digits equal to 8?"
        }
    ],
    temperature=0.2 # Randomness of the output (temp=0 is deterministic)
    # extra_body={"optillm_approach": "rstar"}
)

with open("testRstarOutputGpt4oMini.txt", "w") as f:
    f.write(str(response))

# python optillm.py --rstar-max-depth 6 --rstar-num-rollouts 5 --rstar-c 1.4