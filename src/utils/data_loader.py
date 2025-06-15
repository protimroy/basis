import torch

def xor_data(device='cpu'):
    """Generate XOR dataset on specified device."""
    input_array = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
    output_array = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)
    return (input_array.float(), output_array.float()), \
           (input_array.float(), output_array.float())

def load_data(name, device='cpu'):
    """
    Load dataset by name on specified device.
    
    Args:
        name (str): Name of the dataset ('xor')
        device (str or torch.device): Device to place tensors on
    
    Returns:
        tuple: ((train_inputs, train_labels), (test_inputs, test_labels))
    """
    if name == "xor":
        return xor_data(device)
    else:
        raise ValueError(f"Unknown dataset: {name}")