import yaml

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.load(f)

def save_yaml(yaml_path, data):
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
        