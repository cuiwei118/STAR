import torch

from star import STAR


def get_model(config, device):
    """Build the public STAR model only."""
    if config["MODEL"]["architecture"] != "STAR":
        raise ValueError("This release contains only the STAR architecture.")
    return STAR(config["MODEL"]).to(device)
