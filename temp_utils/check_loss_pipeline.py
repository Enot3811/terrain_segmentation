import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from utils.train_utils import read_config, get_smp_loss_fn


def get_test_cases():
    cases = []
    cases.append({
        'name': 'perfect',
        'targets': torch.tensor([[
            [0, 1, 2, 2],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [2, 2, 1, 0]]]
        ),
        'logits': torch.tensor([[
            [
                [10., 1., 1., 1.],
                [1., 10., 1., 1.],
                [1., 1., 10., 1.],
                [1., 1., 1., 10.]],
            [
                [1., 10., 1., 1.],
                [10., 1., 10., 1.],
                [1., 10., 1., 10.],
                [1., 1., 10., 1.]],
            [
                [1., 1., 10., 10.],
                [1., 1., 1., 10.],
                [10., 1., 1., 1.],
                [10., 10., 1., 1.]]
        ]])
    })

    cases.append({
        'name': 'wrong',
        'targets': torch.tensor([[
            [0, 1, 2, 2],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [2, 2, 1, 0]]]
        ),
        'logits': torch.tensor([[
            [
                [1., 1., 10., 10.],
                [1., 1., 1., 10.],
                [10., 1., 1., 1.],
                [10., 10., 1., 1.]],
            [
                [10., 1., 1., 1.],
                [1., 10., 1., 1.],
                [1., 1., 10., 1.],
                [1., 1., 1., 10.]],
            [
                [1., 10., 1., 1.],
                [10., 1., 10., 1.],
                [1., 10., 1., 10.],
                [1., 1., 10., 1.]]
        ]])
    })

    cases.append({
        'name': 'intermediate',
        'targets': torch.tensor([[
            [0, 1, 2, 2],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [2, 2, 1, 0]]]
        ),
        'logits': torch.tensor([[
            [
                [10., 10., 10., 1.],
                [10., 10., 10., 10.],
                [10., 10., 10., 10.],
                [1., 10., 10., 10.]],
            [
                [1., 1., 1., 1.],
                [10., 1., 1., 1.],
                [1., 10., 1., 1.],
                [1., 1., 1., 1.]],
            [
                [1., 1., 1., 10.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [10., 1., 1., 1.]]
        ]])
    })
    return cases


def test_losses(cases, config_path: Path):
    # Define loss
    config = read_config(config_path)
    loss_fn = get_smp_loss_fn(config['loss'])

    for case in cases:
        logits = case['logits']
        targets = case['targets']
        loss = loss_fn(logits, targets)
        print(f'{loss.__class__.__name__} loss on {case["name"]} case: {loss}')


if __name__ == '__main__':
    cases = get_test_cases()
    config_path = (Path(__file__).parents[1] /
                   'train_configs' / 'test_config.yaml')
    test_losses(cases, config_path)

    
