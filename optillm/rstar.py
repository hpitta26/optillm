import math
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
import re
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state: str, action: str, parent: 'Node' = None, goal: str = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0
        self.goal = goal  # New field to store the goal summary of the problem

def print_tree(node: Node, depth: int = 0):
    indent = "  " * depth
    print(f"{indent}Action: {node.action} | Visits: {node.visits} | Value: {node.value}")
    if depth == 0 and node.goal:
        print(f"{indent}Goal: {node.goal}")
    print(f"{indent}State: {node.state[:100]}...\n")
    for child in node.children:
        print_tree(child, depth + 1)

class RStar:
    def __init__(self, system: str, client, model: str, max_depth: int = 3, num_rollouts: int = 5, c: float = 1.4, actions_to_explore: int = 3):
        self.client = client
        self.model_name = model
        self.max_depth = max_depth
        self.num_rollouts = num_rollouts
        self.c = c
        self.actions = ["A1", "A2", "A3", "A4", "A5"]
        self.original_question = None 
        self.system = system
        self.rstar_completion_tokens = 0
        self.actions_to_explore = actions_to_explore  # Number of actions to explore at each node
        self._current_expanding_node = None  # Track the current node being expanded
        logger.debug(f"Initialized RStar with model: {model}, max_depth: {max_depth}, num_rollouts: {num_rollouts}, actions_to_explore: {actions_to_explore}")

    async def generate_response_async(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate_response, prompt)

    async def expand_async(self, node: Node, action: str) -> Node:
        # Store the current node being expanded so create_prompt can access its goal
        self._current_expanding_node = node
        
        prompt = self.create_prompt(node.state, action)
        new_state = await self.generate_response_async(prompt)
        
        child_node = Node(new_state, action, node)
        if node.goal:  # Propagate the goal to child nodes
            child_node.goal = node.goal
        
        node.children.append(child_node)
        logger.debug(f"Expanded node with action: {action}")
        
        # Clear the reference after expansion is complete
        self._current_expanding_node = None
        
        return child_node

    async def simulate_async(self, node: Node) -> float:
        current_node = node
        depth = 0
        logger.debug("Starting simulation")
        while depth < self.max_depth:
            if not current_node.children:
                allowed_actions = self.allowed_actions_for_node(current_node)
                action = random.choice(allowed_actions)
                current_node = await self.expand_async(current_node, action)
            else:
                current_node = random.choice(current_node.children)
            depth += 1
        value = await self.evaluate_async(current_node)
        logger.debug(f"Simulation complete. Final value: {value}")
        return value

    async def mcts_async(self, root_state: str, goal_summary: str = None) -> List[Node]:
        root = Node(root_state, None, goal=goal_summary)
        tasks = []
        for _ in range(self.num_rollouts):
            tasks.append(self.mcts_rollout_async(root))
        await asyncio.gather(*tasks)

        print('\n\n\n\n\n')
        print_tree(root)  # Recursively print the whole tree
        print('\n\n\n\n\n')

        return self.extract_trajectories(root)

    async def mcts_rollout_async(self, root: Node):
        node = root
        # Selection phase - traverse the tree until we reach a node that hasn't been fully expanded
        while node.children and len(node.children) >= min(len(self.allowed_actions_for_node(node)), self.actions_to_explore):
            node, _ = self.select_action(node)
            
        # Expansion phase - add new children nodes (up to actions_to_explore)
        if len(node.children) < min(len(self.allowed_actions_for_node(node)), self.actions_to_explore):
            allowed_actions = self.allowed_actions_for_node(node)
            # Get actions that haven't been expanded yet
            unexpanded_actions = [a for a in allowed_actions if not any(child.action == a for child in node.children)]
            
            if unexpanded_actions:
                # Select multiple actions to expand (up to actions_to_explore)
                num_to_expand = min(len(unexpanded_actions), self.actions_to_explore - len(node.children))
                actions_to_expand = random.sample(unexpanded_actions, num_to_expand)
                
                # Expand the selected actions
                expansion_tasks = []
                for action in actions_to_expand:
                    expansion_tasks.append(self.expand_async(node, action))
                
                # Wait for all expansions to complete
                expanded_nodes = await asyncio.gather(*expansion_tasks)
                
                # Randomly select one of the newly expanded nodes for simulation
                if expanded_nodes:
                    node = random.choice(expanded_nodes)
            elif node.children:
                # If no unexpanded actions but we have children, select one
                node = random.choice(node.children)
        
        # Simulate and Backpropagate phases
        value = await self.simulate_async(node)
        self.backpropagate(node, value)

    async def solve_async(self, question: str) -> Tuple[str, int]:
        self.original_question = question
        
        # Generate a goal summary for the problem
        goal_summary = await self.generate_goal_summary(question)
        logger.info(f"Generated goal summary: {goal_summary}")
        
        logger.info(f"Solving question: {question}")
        trajectories = await self.mcts_async(question, goal_summary)

        if not trajectories:
            logger.warning("No trajectories found. Unable to solve the question.")
            return "Unable to solve the question due to insufficient reasoning paths.", self.rstar_completion_tokens
        
        final_trajectory = self.select_final_trajectory(trajectories)
        logger.info(f"Total trajectories extracted: {len(trajectories)}")

        # Print each node in the final trajectory to the console
        for i, node in enumerate(final_trajectory):
            print(f"\n\nTrajectory node {i}: | Action: {node.action}\n{node.state}")

        answers = [self.extract_answer(node.state) for node in final_trajectory]
        final_answer = self.select_best_answer(answers)
        logger.info(f"Selected final answer: {final_answer}")
        return final_answer, self.rstar_completion_tokens

    async def generate_goal_summary(self, question: str) -> str:
        """Generate a 3-sentence summary of the problem, its goal, and expected output format."""
        prompt = f"""Given the problem: "{question}"
        
        Please provide a 3-sentence summary that includes:
        1. A concise restatement of the problem
        2. What specifically needs to be solved
        3. What the expected output format should be
        
        Keep your response very brief and to the point.
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a concise mathematical problem summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.2
        )
        self.rstar_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def generate_response(self, prompt: str) -> str:
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a step-by-step mathematical problem solver. Provide ONLY ONE STEP at a time. NEVER provide a complete solution. Focus on providing just ONE incremental insight or calculation. Keep your response under 100 words and NEVER summarize the entire solution."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,  # Limited tokens to prevent complete solutions
            temperature=0.2
        )
        self.rstar_completion_tokens += response.usage.completion_tokens
        generated_response = response.choices[0].message.content.strip()
        logger.debug(f"Generated response: {generated_response}")
        return generated_response
    
    def select_action(self, node: Node) -> Tuple[Node, str]:
        if not node.children:
            allowed_actions = self(node)
            action = random.choice(allowed_actions)
            logger.debug(f"Selected random action: {action}")
            return node, action

        uct_values = []
        for child in node.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = child.value / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
            uct_values.append(uct)

        best_child = node.children[uct_values.index(max(uct_values))]
        logger.debug(f"Selected action: {best_child.action}")
        return best_child, best_child.action

    async def evaluate_async(self, node: Node) -> float:
        """Evaluate the node using OpenAI as the reward model"""
        # Check if we have an answer in the node's state
        answer, conf = self.extract_answer(node.state)
        if answer and conf > 0.5:  # If we have a reasonable answer candidate
            return conf
        
        # Otherwise, use OpenAI to evaluate the quality of the reasoning
        goal = node.goal if node.goal else "Solve the problem step by step"
        
        # Get the path from root to this node
        path = self.get_node_path(node)
        reasoning_steps = "\n".join([f"Step {i+1} ({n.action}): {n.state}" for i, n in enumerate(path[1:])])
        
        prompt = f"""Goal: {goal}
        
        Original question: {self.original_question}
        
        Reasoning steps so far:
        {reasoning_steps}
        
        Evaluate the quality and correctness of these reasoning steps on a scale from 0.0 to 1.0,
        where 0.0 means completely invalid reasoning and 1.0 means perfect reasoning leading to the correct answer.
        Consider factors such as mathematical correctness, relevance to the problem, and progress toward a solution.
        
        Return ONLY a single number between 0.0 and 1.0 representing your evaluation.
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a mathematical reasoning evaluator. Provide a numerical score for the reasoning quality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=64,
            temperature=0.1
        )
        self.rstar_completion_tokens += response.usage.completion_tokens
        
        # Extract the numerical score from the response
        try:
            score_text = response.choices[0].message.content.strip()
            # Use regex to find a number between 0 and 1
            match = re.search(r'([0-9]*\.?[0-9]+)', score_text)
            if match:
                score = float(match.group(1))
                # Ensure the score is between 0 and 1
                score = max(0.0, min(1.0, score))
                logger.debug(f"Evaluated node with OpenAI. Score: {score}")
                return score
            else:
                logger.warning(f"Could not extract numerical score from: {score_text}")
                return 0.5  # Default score
        except Exception as e:
            logger.error(f"Error evaluating node: {e}")
            return 0.5  # Default score in case of error
    
    def get_node_path(self, node: Node) -> List[Node]:
        """Get the path from root to the given node"""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def backpropagate(self, node: Node, value: float):
        logger.debug("Starting backpropagation")
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
        logger.debug("Backpropagation complete")

    def extract_trajectories(self, root: Node) -> List[List[Node]]:
        logger.debug("Extracting trajectories")
        trajectories = []
        stack = [(root, [])]
        while stack:
            node, path = stack.pop()
            if not node.children:
                trajectories.append(path + [node])
            else:
                for child in node.children:
                    stack.append((child, path + [node]))
        logger.debug(f"Extracted {len(trajectories)} trajectories")
        return trajectories

    async def mutual_consistency_async(self, trajectory: List[Node]) -> bool:
        split_index = random.randint(1, len(trajectory) - 1)
        partial_trajectory = trajectory[:split_index]
        prompt = self.create_discriminator_prompt(partial_trajectory)
        completion = await self.generate_response_async(prompt)
        is_consistent = self.compare_completions(completion, trajectory[split_index:])
        print(f"\n\nMutual consistency check: {'Passed' if is_consistent else 'Failed'}\n\n")
        return is_consistent

    def select_final_trajectory(self, trajectories: List[List[Node]]) -> List[Node]:
        logger.debug("Selecting final trajectory")
        # Sort trajectories by the average value per node
        def trajectory_score(traj):
            value_sum = sum(node.value for node in traj)
            return value_sum / len(traj) if len(traj) > 0 else 0
        
        sorted_trajectories = sorted(trajectories, key=trajectory_score, reverse=True)
        return sorted_trajectories[0] if sorted_trajectories else []

    def select_best_answer(self, answers: List[Tuple[str, float]]) -> str:
        valid_answers = [(answer, conf) for answer, conf in answers if answer]
        if not valid_answers:
            return "Unable to determine a valid answer."
        
        # Sort by confidence and then by frequency
        answer_counts = {}
        for answer, conf in valid_answers:
            if answer in answer_counts:
                answer_counts[answer] = (answer_counts[answer][0] + 1, max(answer_counts[answer][1], conf))
            else:
                answer_counts[answer] = (1, conf)
        
        sorted_answers = sorted(answer_counts.items(), key=lambda x: (-x[1][1], -x[1][0]))
        best_answer, (count, conf) = sorted_answers[0]
        
        logger.debug(f"Selected best answer: {best_answer} (count: {count}, confidence: {conf})")
        return best_answer

    def create_prompt(self, state: str, action: str) -> str:
        """
        Create a prompt for the LLM that includes the goal and current state.
        """
        question = self.original_question if hasattr(self, 'original_question') else "the original question"
        
        # Get the goal from the node being expanded
        goal = None
        
        # If this is being called during expand_async, try to find the goal from the current expanding node
        if self._current_expanding_node and hasattr(self._current_expanding_node, 'goal'):
            goal = self._current_expanding_node.goal
        
        # If we couldn't find a goal, use a generic one
        if not goal:
            goal = "Solve the problem in a step-by-step manner to arrive at the correct answer."
        
        # Include the goal as part of each prompt template
        goal_section = f"GOAL: {goal}\n\n"
        
        # Modified prompts that strongly enforce incrementality and include the goal
        prompts = {
        "A1": f"""{goal_section}Given the current state: {state}
PROVIDE ONLY ONE STEP: Generate just the next logical step in solving {question}.
DO NOT restate the problem or provide a complete solution.
Limit your response to ONE insight, calculation, or observation.
Keep it under 100 words. Focus only on what to do next.""",

        "A2": f"""{goal_section}Given the current state: {state}
PROVIDE ONLY ONE STEP: Continue the reasoning with just one single step towards solving {question}.
DO NOT summarize previous work or provide multiple steps.
Focus only on the immediate next calculation or insight.
Keep it under 100 words.""",

        "A3": f"""{goal_section}Given the current state: {state}
PROVIDE ONLY ONE STEP: Identify a single specific sub-question needed to solve {question}.
DO NOT solve the entire problem or recap previous analysis.
Focus on just one sub-question and its immediate answer.
Keep it under 100 words.""",

        "A4": f"""{goal_section}Given the current state: {state}
PROVIDE ONLY ONE STEP: Re-examine only the most recent step using one insight.
DO NOT restate the entire problem or provide a complete solution path.
Focus on clarifying or correcting just one aspect of the previous step.
Keep it under 100 words.""",

        "A5": f"""{goal_section}Given the current state: {state}
PROVIDE ONLY ONE STEP: Rephrase one specific aspect of {question} to aid solving.
DO NOT restate the entire problem or provide a solution approach.
Focus on clarifying just one condition or unknown.
Keep it under 100 words."""
        }
        
        prompt = prompts[action] + "\n\nIMPORTANT: Your response must be ONE SINGLE STEP, not a complete solution. If this single step leads to the answer, you may state 'The answer is [number]' but DO NOT explain the entire solution path again."
        logger.debug(f"Created prompt for action {action}")
        return prompt

    def create_discriminator_prompt(self, partial_trajectory: List[Node]) -> str:
        states = [node.state for node in partial_trajectory]
        partial_reasoning = " ".join(states)
        return f"Given the partial reasoning:\n{partial_reasoning}\nProvide only the next logical step to solve the problem:"

    def compare_completions(self, completion: str, remaining_trajectory: List[Node]) -> bool:
        remaining_states = [node.state for node in remaining_trajectory]
        remaining_reasoning = " ".join(remaining_states)
        
        # Normalize both strings: remove punctuation, convert to lowercase, and split into words
        completion_words = set(completion.lower().replace('.', '').replace(',', '').split())
        trajectory_words = set(remaining_reasoning.lower().replace('.', '').replace(',', '').split())
        
        # Calculate word overlap
        overlap = len(completion_words.intersection(trajectory_words))
        total_words = len(completion_words.union(trajectory_words))
        
        # Consider it a match if there's more than 70% word overlap
        return overlap / total_words > 0.7

    def evaluate(self, node: Node) -> float:
        # Extract the final answer from the node's state
        answer, confidence = self.extract_answer(node.state)
        
        # Check if the answer is a number
        try:
            float(answer)
            logger.debug(f"Evaluated node. Answer: {answer}, Confidence: {confidence}, Value: {confidence}")
            return confidence  # Return the confidence as the value
        except ValueError:
            logger.debug(f"Evaluated node. Answer: {answer}, Confidence: {confidence}, Value: 0.0")
            return 0.0  # If it's not a valid number, return a low score

    def extract_answer(self, final_state: str) -> Tuple[str, float]:
        logger.debug(f"Extracting answer from state: {final_state}")
        patterns = [
            r"The answer is (\d+)",
            r"The final answer is (\d+)",
            r"Therefore, the answer is (\d+)",
            r"So, the answer is (\d+)",
            r"Thus, the answer is (\d+)",
            r"In conclusion, the answer is (\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, final_state)
            if match:
                answer = match.group(1)
                confidence = 1.0
                logger.debug(f"Answer found using pattern '{pattern}': {answer}")
                return answer, confidence
        
        # If no pattern is found, try to extract any number
        numbers = re.findall(r'\d+', final_state)
        if numbers:
            answer = numbers[-1]  # Take the last number found
            confidence = 0.5  # Lower confidence as it's not in the expected format
            logger.debug(f"No pattern found. Using last number as answer: {answer}")
            return answer, confidence
        
        logger.warning("No answer found in the state.")
        return "", 0.0
   
    def solve(self, question: str) -> str:
        """
        Synchronous wrapper for solve_async method.
        """
        return asyncio.run(self.solve_async(question))
    
    def allowed_actions_for_node(self, node: Node) -> List[str]:
        allowed = self.actions.copy()
        # If node is the root, A5 is allowed, but A4 is not
        if node.parent is None:
            if "A4" in allowed:
                allowed.remove("A4")
        else:
            # For non-root nodes, remove A5
            if "A5" in allowed:
                allowed.remove("A5")
            # And if the parent's action is not A3, remove A4
            print("checking node action:", node.parent.action)
            if node.parent.action != "A3":
                allowed.remove("A4")
                print("reached")
        return allowed