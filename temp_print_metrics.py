import json
import os

metrics_path = './comparison_storage/metrics/'
metrics_file_names = os.listdir(metrics_path)

for metrics_file_name in metrics_file_names:
    metrics_file_path = os.path.join(metrics_path, metrics_file_name)
    with open(metrics_file_path, 'r') as file:
        metrics = json.load(file)
    print(metrics_file_name)
    for key, value in metrics.items():
        print(f'{key}: {round(value, 6)}')
    print()
