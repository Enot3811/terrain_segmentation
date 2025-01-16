import yaml
from string import Template


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        # Replace environment variables
        template = Template(f.read())
        yaml_str = template.substitute(os.environ)
        config = yaml.safe_load(yaml_str)
        
    # Process templates
    project_name = config['constants']['project_name']
    version = config['constants']['version']
    
    # Create derived paths
    config['paths']['model_dir'] = f"{project_name}_{version}"
    config['paths']['checkpoint'] = f"{config['paths']['model_dir']}/checkpoint.pth"
    
    return config

with open('train_configs/new_config.yaml', 'r') as file:
    data = yaml.safe_load(file)

print(data)
