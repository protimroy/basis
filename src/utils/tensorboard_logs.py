import os;
import uuid;
from datetime import datetime;

from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

def create_tensorboard_log(log_dir, hyperparams=None, extra_metadata=None):
    """
    Create a TensorBoard log directory with a unique run ID and log hyperparameters/metadata.

    Args:
        log_dir (str): Base directory for TensorBoard logs.
        hyperparams (dict): Hyperparameters to log.
        extra_metadata (dict): Any extra metadata to log.

    Returns:
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        run_id (str): Unique run ID for this experiment.
    """
    # Generate unique run ID and timestamp
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_log_dir = os.path.join(log_dir, run_id)
    os.makedirs(run_log_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(run_log_dir)
    print(f"TensorBoard log directory created at: {run_log_dir}")

    # Log hyperparameters and metadata as text
    if hyperparams is not None:
        writer.add_text("hyperparameters", str(hyperparams))
    if extra_metadata is not None:
        writer.add_text("metadata", str(extra_metadata))
    writer.add_text("run_id", run_id)
    writer.add_text("timestamp", datetime.now().isoformat())

    return writer, run_id


def log_scalar(writer, tag, value, step):
    """
    Log a scalar value to TensorBoard.

    Args:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        tag (str): Tag for the scalar value.
        value (float): Scalar value to log.
        step (int): Global step value to record with the scalar.
    """
    writer.add_scalar(tag, value, step)
    print(f"Logged scalar '{tag}' with value {value} at step {step}")

def log_histogram(writer, tag, values, step):
    """
    Log a histogram of values to TensorBoard.

    Args:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        tag (str): Tag for the histogram.
        values (torch.Tensor): Values to log as a histogram.
        step (int): Global step value to record with the histogram.
    """
    # Values should already be a torch.Tensor if numpy is not used
    writer.add_histogram(tag, values, step)
    print(f"Logged histogram '{tag}' at step {step}")

def close_tensorboard_log(writer):
    """
    Close the TensorBoard SummaryWriter.

    Args:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance to close.
    """
    writer.close()
    print("TensorBoard SummaryWriter closed.")