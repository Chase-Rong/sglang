"""Validate completed requests and cross-check Decode batches against ready intervals."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('summary', type=Path)
args = parser.parse_args()
data = json.loads(args.summary.read_text())
assert not data['errors'] and data['completed'] == data['submitted']
events, accept_lengths = [], []
for row in data['requests']:
    meta = row['meta_info']
    assert row['prompt_tokens'] == meta['prompt_tokens'] == 262144
    assert row['completion_tokens'] == meta['completion_tokens'] == 1024
    assert meta.get('num_retractions', 0) == 0
    assert meta['spec_num_proposed_drafts'] > 0 and meta['spec_accepted_drafts'] > 0
    assert meta['prefill_finished_time'] < meta['request_finished_ts']
    events.extend([(meta['prefill_finished_time'], 1), (meta['request_finished_ts'], -1)])
    accept_lengths.append(meta['spec_accept_length'])
current = maximum = 0
for time, delta in sorted(events):
    current += delta
    maximum = max(current, maximum)
assert current == 0
graph_rows = [row for row in data['decode_evidence'] if row['full_context'] and row['graph']]
report = {
    'submitted': data['submitted'], 'completed': data['completed'],
    'input_tokens_per_request': 262144, 'output_tokens_per_request': 1024,
    'max_simultaneous_after_prefill': maximum,
    'max_full_context_decode': data['max_full_context_decode'],
    'max_full_context_graph_decode': data['max_full_context_graph_decode'],
    'max_batch_graph_log_samples': sum(row['running'] == data['max_full_context_graph_decode'] for row in graph_rows),
    'mtp_accept_length_range': [min(accept_lengths), max(accept_lengths)],
    'retractions': 0, 'errors': [], 'elapsed_s': data['elapsed_s'],
    'peak_graph_evidence': [row['line'] for row in graph_rows if row['running'] == data['max_full_context_graph_decode']],
}
output = args.summary.parent / 'ready_capacity_verified.json'
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
