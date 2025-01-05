import random
from powhf import utils

SEED = 0


def subsample_data(data, subsample_size):
    """Subsamples data."""
    utils.debug_log(
        f"powhf.data.subsample_data :: Subsampling data, size: {subsample_size}"
    )
    random.seed(SEED)
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    utils.debug_log(
        f"powhf.data.subsample_data :: Subsampled data: {len(inputs)} samples"
    )
    return inputs, outputs


def create_split(data, split_size):
    """Splits data into two sets."""
    utils.debug_log(
        f"powhf.data.create_split :: Splitting data, split size: {split_size}"
    )
    random.seed(SEED)
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    utils.debug_log(
        f"powhf.data.create_split :: Split data, part1: {len(inputs1)}, part2: {len(inputs2)}"
    )
    return (inputs1, outputs1), (inputs2, outputs2)
