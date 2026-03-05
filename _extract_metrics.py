import torch, json

b1 = torch.load('results/baseline_history.pt',       map_location='cpu', weights_only=True)['history']
d1 = torch.load('results/displacement_history.pt',   map_location='cpu', weights_only=True)['history']
b2 = torch.load('results/2d_baseline_history.pt',    map_location='cpu', weights_only=True)['history']
d2 = torch.load('results/2d_displacement_history.pt',map_location='cpu', weights_only=True)['history']

data = {
    "1d": {
        "baseline_l2":     b1["l2_log"][-1],
        "disp_l2":         d1["l2_log"][-1],
        "gate_final":      d1["gate_rate_log"][-1],
        "gate_sparsity":   (1 - d1["gate_rate_log"][-1]) * 100,
        "l2_improvement":  (b1["l2_log"][-1] - d1["l2_log"][-1]) / b1["l2_log"][-1] * 100,
        "time_baseline":   b1["wall_time_log"][-1],
        "time_disp":       d1["wall_time_log"][-1],
        "epochs":          b1["epochs_log"][-1],
    },
    "2d": {
        "baseline_l2":     b2["l2_log"][-1],
        "disp_l2":         d2["l2_log"][-1],
        "gate_final":      d2["gate_rate_log"][-1],
        "gate_sparsity":   (1 - d2["gate_rate_log"][-1]) * 100,
        "l2_improvement":  (b2["l2_log"][-1] - d2["l2_log"][-1]) / b2["l2_log"][-1] * 100,
        "time_baseline":   b2["wall_time_log"][-1],
        "time_disp":       d2["wall_time_log"][-1],
        "epochs":          b2["epochs_log"][-1],
    }
}

with open("results/experiment_summary.json", "w") as f:
    json.dump(data, f, indent=2)

print(json.dumps(data, indent=2))
