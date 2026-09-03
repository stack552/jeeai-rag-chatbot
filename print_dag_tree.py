from dag_cache import get_node, get_children, ROOT_NODE_ID

def print_tree(node_id, depth=0):
    node = get_node(node_id)
    if node is None:
        print("  " * depth + f"[missing node: {node_id}]")
        return

    if node_id == ROOT_NODE_ID:
        label = "ROOT"
    else:
        short_id = node_id[:8]
        label = f"{node['question']!r}  [id={short_id}...]"

    print("  " * depth + f"- {label}")

    for child in get_children(node_id):
        print_tree(child["id"], depth + 1)

print_tree(ROOT_NODE_ID)