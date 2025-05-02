import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.enrich_taxonomy import traverse_and_add_children, load_json_from_file, max_depth
import json

# File paths
input_path = "demo_example/example_taxonomy.json"
output_path = "demo_example/example_taxonomy_enriched.json"

# Load input taxonomy
taxonomy = load_json_from_file(input_path)

# Enrich it
depth = max_depth(taxonomy) - 1
print(f"Max depth of the tree: {depth}")

taxonomy = traverse_and_add_children(taxonomy, max_depth=depth)

# Save output
with open(output_path, "w") as f:
    json.dump(taxonomy, f, indent=4)

print(f"Saved enriched taxonomy to: {output_path}")
