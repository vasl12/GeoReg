import os
from datetime import datetime
import torch
import yaml


def normalize(x):
    min_ = x.min()
    max_ = x.max()
    return (x - min_) / (max_ - min_)


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(len(dims))]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    # coordinate_tensor = coordinate_tensor.view([torch.prod(torch.tensor(dims)), len(dims)])

    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_coordinate_tensor_like(x, channel_first=True, indexing="ij"):
    if channel_first:
        spatial_dims = x.shape[2:]  # Channels are always on dim 1, coordinates are on -1
    else:
        spatial_dims = x.shape[1:-1]
    coordinate_tensor = [torch.linspace(-1., 1., i, dtype=torch.float32, device=x.device)
                         for i in spatial_dims]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing=indexing)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=-1)
    coordinate_tensor = coordinate_tensor.tile((x.shape[0], *[1] * len(coordinate_tensor.shape)))
    return coordinate_tensor


def calculate_distances(p1, p2):
    assert len(p1.shape) == len(p2.shape)
    assert p1.shape[-1] == p2.shape[-1]
    dists = p1.unsqueeze(-2) - p2.unsqueeze(-3)
    dists = dists * dists
    dists = dists.sum(-1)
    return dists


def calculate_closest_indices(c1, c2, round_to_closest=True):
    # TODO: double check
    """ We assume the c1 coordinates live in the coordinate space of c2.
    c1 is mapped into the index space of c2, then round and convert to torch.long to obtain
    which c2 indices each c1 coordinate should map to. """
    # We assume the (-1, -1) index will point us to how far it is from the [1.0, 1.0] edge coordinate of the image
    c2_corner = c2[0, -1, -1] if len(c2.shape[1:-1]) == 2 else c2[0, -1, -1, -1]
    c1_neg1to1 = c1 * (1.0 / c2_corner)  # Normalize to [-1.0, 1.0]
    c1_0to1 = (c1_neg1to1 + 1) / 2  # Normalize to [0.0, 1.0]
    c2_shape = torch.tensor(c2.shape[1:-1], device=c1.device)
    c1_idx_space = c1_0to1 * (c2_shape - 1.0)  # Map to c2 index space
    c1_idx_space_clipped = c1_idx_space.clip(min=torch.zeros_like(c2_shape), max=(c2_shape - 1.0))
    if round_to_closest:
        c1_idx_space_clipped = torch.round(c1_idx_space_clipped)  # Round to closest index, then to integer
    c1_idx = c1_idx_space_clipped.to(torch.long)
    return c1_idx



def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def read_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print(e)


# Function to get current date and time formatted as specified
def get_current_datetime():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    return date_str, time_str


def create_ckpt_directory(checkpoints_dir):

    # Create directory for checkpoints
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Get current date and time
    current_date, current_time = get_current_datetime()

    # Create directory with current date if it doesn't exist
    date_dir = os.path.join(checkpoints_dir, current_date)
    os.makedirs(date_dir, exist_ok=True)

    # Create directory with current time inside the date directory
    current_dir = os.path.join(date_dir, current_time)
    os.makedirs(current_dir, exist_ok=True)

    print("Created checkpoint directory:", current_dir)

    return current_dir



