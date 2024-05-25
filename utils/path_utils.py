import dataclasses
import socket
import os


def get_computer_id():
    hostname = socket.gethostname()
    if hostname == HOSTNAME:
        return hostname
    else:
        raise Exception(f'Unknown hostname: {hostname}.')


@dataclasses.dataclass
class PathHolder:
    dataset_folder: str
    processed_folder: str
    log_folder: str


def get_data_paths(dset: str):
    computer_id = get_computer_id()
    if computer_id == HOSTNAME:
        dset_paths = {"brain_camcan": None,
                      "brain_camcan2d": None,
                      "nlst": None,
                      "mnist": None,
                      "rafd": None,
                      }
        # You can also add custom paths for your /home/ directory
        user = computer_id[:-4]
            # dset_paths["dset_name"] = f"/u/home/{user}/path_to_dset"
    else:
        raise Exception(f'Undefined data paths for computer_id: {computer_id}')
    return dset_paths[dset]
