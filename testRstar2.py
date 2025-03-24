import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from optillm.rstar import RStar, Node, print_tree
load_dotenv()

# Assume your Node, RStar, and print_tree are defined above or imported from your module.
# For example:
# from rstar_module import Node, RStar, print_tree

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = "http://localhost:8000/v1"
client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)

async def main():
    question = "For how many two-digit primes is the sum of the digits equal to 8?"
    
    # Create an instance of RStar using the rstar approach.
    rstar_instance = RStar(system="test", client=client, model="rstar-gpt-4o", max_depth=3, num_rollouts=5, c=1.4)
    
    # Instead of calling the completions endpoint directly, we run the MCTS process.
    root = Node(question, None)
    
    # Run a single rollout for demonstration (you could run more)
    await rstar_instance.mcts_rollout_async(root)
    
    # Now print the entire tree starting from the root node.
    print_tree(root)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
