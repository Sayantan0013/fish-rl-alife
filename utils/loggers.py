import h5py
import torch

def log_step_to_hdf5(file_path, run_idx, step, data_dict):
    """
    data_dict: dictionary of tensors or numpy arrays
    """
    with h5py.File(file_path, "a") as f:
        run_group = f.require_group(f"run_{run_idx}")
        step_group = run_group.create_group(f"step_{step}")

        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()  # convert tensor to numpy
            step_group.create_dataset(key, data=value)


def load_hdf5_log(file_path, run_idx=None, step=None):
    """
    Load data from the HDF5 log.

    Args:
        file_path (str): Path to the HDF5 file.
        run_idx (int, optional): If specified, load only this run.
        step (int, optional): If specified, load only this step.

    Returns:
        dict: Nested dictionary {run_idx: {step: {key: value}}}
    """
    data = {}
    with h5py.File(file_path, "r") as f:
        runs = [f"run_{run_idx}"] if run_idx is not None else list(f.keys())

        for run in runs:
            data[run] = {}
            steps = [f"step_{step}"] if step is not None else list(f[run].keys())

            for st in steps:
                data[run][st] = {}
                for key, ds in f[run][st].items():
                    data[run][st][key] = np.array(ds)

    return data
