import json

with open('experiments/2026-02-02_19-19-34_gpu_ffa_pulsar_attempt_0/logs/0-run/stage_1_initial_implementation_1_preliminary/journal.json', 'r') as f:
    data = json.load(f)

for i, node in enumerate(data['nodes']):
    if node.get('exc_type'):
        print(f"Node {i} (ID: {node['id']}):")
        print(f"  Exc Type: {node['exc_type']}")
        print(f"  Exc Info: {node['exc_info']}")
        print(f"  Plan: {node.get('plan', 'No plan')[:100]}...")
        print("-" * 20)
