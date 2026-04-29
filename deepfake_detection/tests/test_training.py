from src.training.losses import CompositeLoss


def test_composite_loss_init():
    criterion = CompositeLoss()
    assert criterion.bce_weight == 1.0
