from src.losses import Loss
from src.model import Model


def split_batch(data, target, batch_size):
    for i in range(0, len(data), batch_size):
        if i + batch_size < len(data):
            yield data[i:i + batch_size], target[i:i + batch_size]
        else:
            yield data[i:], target[i:]


def train_step(data, target, model: Model, loss_fn: Loss, learning_rate):
    pred = model(data)
    loss = loss_fn(pred, target)
    grad = loss_fn.backward()
    model.backward(grad)
    model.update(learning_rate=learning_rate)
    return loss


def test_step(data, target, model: Model, loss_fn: Loss):
    pred = model(data)
    loss = loss_fn(pred, target)
    return loss
