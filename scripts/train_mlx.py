"""Use the MLX-LM CLI with an explicit answer-only, padding-free loss."""


def target_mask(lengths, target_width, array_module):
    positions = array_module.arange(1, target_width + 1)[None, :]
    return (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:2])


def answer_only_loss(model, batch, lengths):
    import mlx.core as mx
    import mlx.nn as nn
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    mask = target_mask(lengths, targets.shape[1], mx)
    count = mask.sum()
    loss = nn.losses.cross_entropy(logits, targets) * mask
    return loss.astype(mx.float32).sum() / count, count


def main():
    from functools import partial
    from mlx_lm import lora
    # The trainer exposes a loss argument and uses it for validation as well.
    lora.train = partial(lora.train, loss=answer_only_loss)
    lora.evaluate = partial(lora.evaluate, loss=answer_only_loss)
    lora.main()


if __name__ == '__main__':
    main()
