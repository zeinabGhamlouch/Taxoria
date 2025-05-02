import llama_index
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.llms import Ollama
import json
import re
from typing import Dict, List

# Load Hugging Face embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# load llm
llm = Ollama(model="llama3")

def max_depth(node: Dict) -> int:
    """
    Recursively calculates the maximum depth of the tree.
    """
    if not node.get("children"):
        return 1
    return 1 + max(max_depth(child) for child in node["children"])

def deduplicate_list(items: List[Dict]) -> List[Dict]:
    """
    Deduplicates a list of dictionaries by ensuring unique 'name' keys.
    If duplicates exist, the one with 'original' source is kept.
    """
    seen = {}
    for item in items:
        name = item["name"]
        if name not in seen or seen[name]["source"] != "original":
            seen[name] = item
    return list(seen.values())

def ensure_source_recursive(node: Dict, default_source: str = "original-taxonomy") -> Dict:
    """
    Ensures all nodes at all depths have a 'source' key, and the 'source' key
    is placed after the 'name' key.
    """
    if "source" not in node:
        node["source"] = default_source

    ordered_node = {key: node[key] for key in ["name", "source"] if key in node}

    for key in node:
        if key not in ordered_node:
            ordered_node[key] = node[key]

    if "children" in node:
        ordered_node["children"] = [ensure_source_recursive(child, default_source) for child in node["children"]]

    return ordered_node

def convert_names_to_children(name_list: List[str]) -> List[Dict]:
    """
    Converts a list of child names into a list of dictionaries with 'name' keys and an empty 'children' list.
    Each new child gets source as 'llm-generated'.
    """
    return [{"name": name, "source": "llm-generated", "children": []} for name in name_list]

def extract(text: str) -> List[str]:
    """
    Extract categories from the input text.
    """
    extracted_categories = []
    text = text.strip()
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        categories = re.findall(r'\d+\.\s(.+)', line)
        if not categories:
            categories = re.findall(r'\*\s*(.+)', line)
        if not categories:
            categories = re.findall(r'-\s*(.+)', line)

        categories = [item.lstrip('*').strip() for item in categories]
        extracted_categories.extend(categories)

def calculate_similarity(word1: str, word2: str) -> float:
    """Calculate cosine similarity using Hugging Face embeddings."""
    vec1 = embed_model.get_text_embedding(word1)
    vec2 = embed_model.get_text_embedding(word2)

    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

def filter_children_by_score(parent_name: str, children: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """
    Filters out children that do not meet the similarity threshold, removing the similarity_score from the final result.
    Also ensures that all new children have "source" set to "llm-generated".
    """
    filtered_children = []
    for child in children:
        score = calculate_similarity(parent_name, child["name"])
        print(f"Similarity between '{parent_name}' and '{child['name']}': {score:.4f}")
        if score is not None and score >= threshold:
            child["similarity_score"] = score
            # Ensure that the source is "llm-generated"
            child["source"] = "llm-generated"
            filtered_children.append({key: value for key, value in child.items() if key != "similarity_score"})
    return filtered_children

def add_children_to_node_with_score(tree: Dict, target_node: str, new_children: List[Dict], threshold: float = 0.9) -> Dict:
    """
    Adds new children to a node with the specified name while filtering them by similarity score.
    Ensures that the new children have source 'llm-generated'.
    """
    if not isinstance(tree, dict):
        return tree  # If not a dictionary, return as-is

    tree = ensure_source_recursive(tree)

    # Check if this is the target node
    if tree.get("name") == target_node:
        existing_children = tree.get("children", [])
        new_children_filtered = filter_children_by_score(target_node, new_children, threshold)
        # Merge children correctly, without overwriting previous ones
        tree["children"] = deduplicate_list(existing_children + new_children_filtered)  # Merge and remove duplicates
        return tree

    # If this node has children, search recursively
    if "children" in tree:
        tree["children"] = [add_children_to_node_with_score(child, target_node, new_children, threshold) for child in tree["children"]]

    return tree  # Return updated tree

def traverse_and_add_children(tree: Dict, current_depth: int = 0, max_depth: int = 6, threshold: float = 0.9) -> Dict:
    """
    Traverse the tree and for each node, fetch its direct children and add them recursively.
    Stops at the specified maximum depth.
    """
    if current_depth > max_depth:
        return tree  # Stop recursion if we reach the max depth

    if not isinstance(tree, dict):
        return tree

    # Process the current node (add its children)
    node_name = tree.get("name")
    if node_name:
        prompt = f'''As an LLM, give me the top 3 subclasses of {node_name}
                    Answer by the subclasses only without any justification'''
        directs = llm.invoke(prompt)
        print(f"Direct children of {node_name}")
        directs = extract(directs)
        directs = convert_names_to_children(directs)
        tree = add_children_to_node_with_score(tree, node_name, directs, threshold)
        print(f"Updated tree for node {node_name}: {tree}")

    # Recurse into children if they exist
    if "children" in tree:
        tree["children"] = [traverse_and_add_children(child, current_depth + 1, max_depth, threshold) for child in tree["children"]]
        return tree


def load_json_from_file(file_path: str) -> Dict:
    """
    Loads a JSON file and returns the content as a dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


