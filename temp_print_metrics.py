import json
import os

TEST_ONLY = True

metrics_path = './comparison_storage/metrics/'
metrics_file_names = os.listdir(metrics_path)
if TEST_ONLY:
    metrics_file_names = filter(lambda file: '_test_' in file, metrics_file_names)

for metrics_file_name in metrics_file_names:
    metrics_file_path = os.path.join(metrics_path, metrics_file_name)
    with open(metrics_file_path, 'r') as file:
        metrics = json.load(file)
    print(metrics_file_name)
    for key, value in metrics.items():
        print(f'{key}: {round(value, 6)}')
    print()
