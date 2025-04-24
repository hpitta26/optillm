import os
import asyncio
import matplotlib
# Force Agg backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
import logging
from pathlib import Path
import textwrap

# Import your RStar implementation
from optillm.rstar import RStar, Node, print_tree

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# Enhanced tree printing function that shows state content and goal
def enhanced_print_tree(node: Node, depth: int = 0, max_chars: int = 1000):
    """
    Print the tree with node state content (truncated for readability)
    """
    indent = "  " * depth
    action = node.action if node.action else "Root"
    
    # Truncate state for display
    if node.state:
        display_state = node.state[:max_chars] + "..." if len(node.state) > max_chars else node.state
        display_state = display_state.replace("\n", " ")
    else:
        display_state = "No state"
    
    print(f"{indent}Node: {action} | Visits: {node.visits} | Value: {node.value:.2f}")
    if depth == 0 and node.goal:
        print(f"{indent}Goal: {node.goal}")
    print(f"{indent}State: {display_state}")
    print(f"{indent}{'-' * 40}")
    
    for child in node.children:
        enhanced_print_tree(child, depth + 1, max_chars)

def create_incremental_mcts_visualization(root_node, filename="mcts_incremental_steps.png", num_actions_to_show=3):
    """
    Create a visualization of the MCTS tree showing actual incremental steps.
    
    Parameters:
    - root_node: The root node of the MCTS tree
    - filename: Output filename for the visualization
    - num_actions_to_show: Number of explored actions to display at each level
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Define colors for different actions
    action_colors = {
        None: '#E8E8E8',  # Light gray for root
        'A1': '#AED6F1',  # Blue
        'A2': '#A3E4D7',  # Green
        'A3': '#F9E79F',  # Yellow
        'A4': '#F5B7B1',  # Red
        'A5': '#D7BDE2'   # Purple
    }
    
    all_actions = ['A1', 'A2', 'A3', 'A4', 'A5']
    max_depth = 15  # Maximum depth to visualize
    
    # Counter for node IDs
    node_counter = 0
    
    # Dictionary to track existing nodes and their IDs
    visited_nodes = {}
    
    # Helper function to get a concise step label for each node
    def get_step_label(node):
        if not node.state:
            return "No step"
        
        # For the root node, include the goal if available
        if not node.action:
            if node.goal:
                # Show first sentence of goal
                first_sentence = node.goal.split('.')[0] + '.' if '.' in node.goal else node.goal
                if len(first_sentence) > 50:
                    return first_sentence[:50] + '...'
                return first_sentence
            else:
                words = node.state.split()
                if len(words) > 10:
                    return ' '.join(words[:10]) + '...'
                return node.state
        
        # For action nodes, use action-specific prefixes
        action_prefixes = {
            'A1': "Step: ",
            'A2': "Next: ",
            'A3': "Sub-Q: ",
            'A4': "Review: ",
            'A5': "Restate: "
        }
        
        prefix = action_prefixes.get(node.action, "")
        
        # Get a single-line summary (first 1-2 sentences or first 15 words)
        text = node.state.strip()
        
        # Try to get the first sentence
        if '.' in text:
            first_sentence = text.split('.')[0] + '.'
            if len(first_sentence.split()) > 40:
                words = text.split()[:40]
                return prefix + ' '.join(words) + '...'
            return prefix + first_sentence
        
        # Otherwise just take the first few words
        words = text.split()
        if len(words) > 40:
            return prefix + ' '.join(words[:40]) + '...'
        return prefix + text
    
    # Helper function to sort child nodes by visits and value (to prioritize most explored paths)
    def sort_children_by_importance(children):
        # Sort by visits (descending) and then by value (descending)
        return sorted(children, key=lambda x: (x.visits, x.value), reverse=True)
    
    # Build the complete tree structure with the top N explored actions
    def build_complete_tree(node, node_id, depth=0):
        nonlocal node_counter
        
        # Get step label
        step_label = get_step_label(node)
        
        # Create node label with action, value, and content
        action = node.action if node.action else "Root"
        label = f"{action}\nVisits: {node.visits}\nValue: {node.value:.2f}\n\n{step_label}"
        
        # For root node, also include the goal if available
        if depth == 0 and node.goal:
            # Add a wrapped version of the goal to the label
            wrapped_goal = textwrap.fill(f"Goal: {node.goal}", width=40)
            label = f"{label}\n\n{wrapped_goal}"
        
        # Add node to graph
        G.add_node(
            node_id, 
            label=label,
            color=action_colors.get(node.action, '#E8E8E8'),
            shape='ellipse',
            style='filled',
            depth=depth
        )
        
        # Don't expand beyond max depth
        if depth >= max_depth:
            return
        
        # Get allowed actions for this node
        rstar_instance = RStar(system="", client=None, model="")
        allowed_actions = rstar_instance.allowed_actions_for_node(node)
        
        # Sort children by visits and value to identify most important paths
        sorted_children = sort_children_by_importance(node.children)
        
        # Map of child nodes by action (for most visited children)
        top_children = sorted_children[:num_actions_to_show] if len(sorted_children) > num_actions_to_show else sorted_children
        top_actions = [child.action for child in top_children]
        child_by_action = {child.action: child for child in top_children}
        
        # Keep track of which actions were explored vs not explored
        explored_actions = set(child.action for child in node.children)
        
        # Add all allowed actions as children
        for action in allowed_actions:
            # If this is a top child (one of the most visited)
            if action in top_actions:
                child_node = child_by_action[action]
                # Check if we've already created this node
                if child_node in visited_nodes:
                    child_id = visited_nodes[child_node]
                else:
                    node_counter += 1
                    child_id = f"node_{node_counter}"
                    visited_nodes[child_node] = child_id
                    # Recursively build tree for this child
                    build_complete_tree(child_node, child_id, depth + 1)
                
                # Add an edge with solid line for explored paths
                G.add_edge(
                    node_id, 
                    child_id, 
                    style='solid', 
                    weight=2.0 + (child_node.visits / 10),  # Thicker edges for more visited nodes
                    alpha=1.0
                )
            elif action in explored_actions:
                # This is an explored action but not one of the top ones we want to visualize
                # We'll create a simplified node without recursing further
                child_node = next(child for child in node.children if child.action == action)
                node_counter += 1
                child_id = f"node_{node_counter}"
                
                # Create label for less important but explored node
                step_label = get_step_label(child_node)
                label = f"{action}\nVisits: {child_node.visits}\nValue: {child_node.value:.2f}\n\n{step_label}"
                
                G.add_node(
                    child_id,
                    label=label,
                    color=action_colors.get(action, '#E8E8E8'),
                    alpha=0.7,  # Slightly transparent
                    shape='ellipse',
                    style='filled',
                    depth=depth + 1
                )
                
                # Add edge with medium weight
                G.add_edge(
                    node_id, 
                    child_id, 
                    style='solid', 
                    weight=1.5,
                    alpha=0.8
                )
            else:
                # Create a placeholder node for unexplored action
                node_counter += 1
                child_id = f"node_{node_counter}"
                
                # Add placeholder node with dimmed appearance
                G.add_node(
                    child_id,
                    label=f"{action}\nNot Explored",
                    color=action_colors.get(action, '#E8E8E8'),
                    alpha=0.3,  # Make it transparent
                    shape='ellipse',
                    style='dashed',
                    depth=depth + 1
                )
                
                # Add dashed edge for unexplored paths
                G.add_edge(
                    node_id, 
                    child_id, 
                    style='dashed', 
                    weight=0.7,
                    alpha=0.3
                )
    
    # Start building the tree from the root
    visited_nodes[root_node] = "node_0"
    build_complete_tree(root_node, "node_0")
    
    # Create the figure
    plt.figure(figsize=(24, 18))
    
    # Create a hierarchical layout optimized for trees
    try:
        # Try to use dot layout for better tree visualization
        pos = nx.drawing.nx_agraph.graphviz_layout(
            G, 
            prog='dot',
            args='-Grankdir=TB -Gnodesep=1.5 -Granksep=2.0'
        )
    except Exception as e:
        logger.warning(f"Graphviz layout failed: {e}, falling back to multipartite layout")
        
        # Group nodes by depth level for multipartite layout
        layer_groups = {}
        for node, attrs in G.nodes(data=True):
            depth = attrs.get('depth', 0)
            if depth not in layer_groups:
                layer_groups[depth] = []
            layer_groups[depth].append(node)
        
        # Create a multipartite layout based on depth
        pos = nx.multipartite_layout(G, subset_key='depth', align='vertical')
    
    # Draw nodes - bigger nodes to fit text
    for node, attrs in G.nodes(data=True):
        alpha = attrs.get('alpha', 0.8)
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[node],
            node_color=attrs.get('color', '#E8E8E8'),
            node_size=5000,  # Medium size nodes for incremental content
            alpha=alpha,
            edgecolors='black',
            linewidths=1.0
        )
    
    # Draw edges with style attributes
    for u, v, attrs in G.edges(data=True):
        edge_style = attrs.get('style', 'solid')
        edge_width = attrs.get('weight', 1.0)
        edge_alpha = attrs.get('alpha', 1.0)
        
        if edge_style == 'dashed':
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=edge_width,
                alpha=edge_alpha,
                style='dashed',
                arrows=True,
                arrowsize=15,
                connectionstyle='arc3,rad=0.1'
            )
        else:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=edge_width,
                alpha=edge_alpha,
                arrows=True,
                arrowsize=20,
                connectionstyle='arc3,rad=0.1'
            )
    
    # Draw labels with node content
    nx.draw_networkx_labels(
        G, pos,
        labels={node: attrs['label'] for node, attrs in G.nodes(data=True)},
        font_size=10,
        font_family='sans-serif',
        font_weight='bold',
        verticalalignment='center',  # Center text vertically
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=4.0)  # Text background for readability
    )
    
    # Create a legend for action types
    legend_patches = [
        mpatches.Patch(color=action_colors[action], label=action if action else "Root")
        for action in [None] + all_actions
    ]
    # Add legend entries for explored vs unexplored nodes
    legend_patches.append(mpatches.Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Not Explored'))
    legend_patches.append(mpatches.Patch(facecolor='gray', alpha=0.8, edgecolor='black', label='Explored'))
    
    plt.legend(handles=legend_patches, loc='upper right', fontsize=12)
    
    # Set title and adjust layout
    plt.title("MCTS Tree with Incremental Steps", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout(pad=2.0)
    
    # Save the figure with high resolution
    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"MCTS tree with incremental steps saved to {output_path}")
    
    # Close the figure to prevent memory leaks
    plt.close()
    
    return output_path

async def main():
    # The math question to solve
    question = "For how many two-digit primes is the sum of the digits equal to 8?"
    
    logger.info(f"Starting MCTS analysis for question: {question}")
    
    # Create RStar instance with custom system prompt to encourage step-by-step reasoning
    rstar_instance = RStar(
        system="You are a step-by-step mathematical problem solver. Provide ONLY ONE STEP at a time. NEVER provide a complete solution. Focus on providing just ONE incremental insight or calculation. Keep your response under 100 words and NEVER summarize the entire solution.",
        client=client,
        model="gpt-4o",
        max_depth=3,
        num_rollouts=5,
        c=1.4,
        actions_to_explore=3  # Now using the new parameter to explore 3 actions
    )
    
    # Set the original question
    rstar_instance.original_question = question
    
    # Initialize root node
    root = Node(state=question, action=None)
    
    # Run multiple MCTS rollouts
    rollouts = 3
    for i in range(rollouts):
        logger.info(f"Running MCTS rollout {i+1}/{rollouts}")
        await rstar_instance.mcts_rollout_async(root)
    
    # Print enhanced tree structure to console
    print("\n================ MCTS TREE STRUCTURE ================")
    enhanced_print_tree(root)
    print("=====================================================\n")
    
    # Create a visualization with incremental node content
    # Pass the num_actions_to_show parameter to display 3 actions at each level
    output_path = create_incremental_mcts_visualization(root, "mcts_incremental_steps.png", num_actions_to_show=3)
    print(f"\nMCTS tree with incremental steps saved to: {output_path}")
    
    # Get final answer
    try:
        answer, tokens = await rstar_instance.solve_async(question)
        print(f"\nFinal answer: {answer}")
        print(f"Total tokens used: {tokens}")
    except Exception as e:
        logger.error(f"Error getting final answer: {e}")
        print(f"\nError getting final answer: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()