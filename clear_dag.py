from dag_cache import r, NODE_KEY_PREFIX, ensure_root_exists

deleted_count = 0
for key in r.scan_iter(match=f"{NODE_KEY_PREFIX}*"):
    r.delete(key)
    deleted_count += 1

print(f"Deleted {deleted_count} DAG node(s).")

# Recreate a fresh, empty root
ensure_root_exists()

# Force Redis to write this state to disk immediately, so it survives
# a container restart. Without this, if Redis stops before its next
# automatic snapshot, a restart would reload the OLD data, undoing
# this deletion entirely.
r.save()
print("Saved to disk. Fresh ROOT node created. DAG is now empty and persisted.")