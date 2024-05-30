import json

data = None
with open('/home/thomasl/tmdt-benchmark/data/class-01/old.json', 'r') as inputf:
    data = json.load(inputf)

for i in range(len(data['frames'])):
    tmp = data['frames'][i]['file_path']
    data['frames'][i]['file_path'] = f"train/{tmp[:-4]}"
    tmp_matrix = data['frames'][i]['transform_matrix'][0]
    data['frames'][i]['transform_matrix'] = tmp_matrix

with open('/home/thomasl/tmdt-benchmark/data/class-01/out-transforms.json', 'w') as outputf:
    json.dump(data, outputf, indent=2)