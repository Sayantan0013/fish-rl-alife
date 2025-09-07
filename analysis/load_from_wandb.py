import wandb
import yaml
import tempfile
import os

def load_yaml_from_wandb_run(project_name, run_id, file_name = 'config.yaml'):
    """
    Load a YAML file from a wandb run and return it as a dictionary.

    Args:
        project_name (str): Name of the wandb project
        run_id (str): ID of the wandb run
        yaml_filename (str): Name of the YAML file to load

    Returns:
        dict: Contents of the YAML file as a dictionary
    """
    # Initialize wandb API
    api = wandb.Api()


    # Get the run
    run = api.run(f"{project_name}/{run_id}")

    # Create a temporary directory to download the file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the YAML file
        file_path = os.path.join(temp_dir, file_name)
        run.file(file_name).download(root=temp_dir, replace=True)

        # Load and parse the YAML file
        with open(file_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)

    return yaml_dict

if __name__ == "__main__":
    # Example usage:
    config_dict = load_yaml_from_wandb_run(
        project_name="fish-marl-cur-experiment",
        run_id="2m5mtesa",
    )
