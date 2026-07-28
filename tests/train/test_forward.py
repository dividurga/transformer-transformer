import logging

import hydra
import torch
from omegaconf import OmegaConf

from t2.model.t2 import create_random_batch
from t2.train.utils import set_torch_configs

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))


def test_t2_forward(num_steps: int = 10, num_repeat_batch: int = 4):
    """Test T2 model forward pass with the test configuration."""
    logging.info("Starting T2 forward pass test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps"
    logging.info(f"Using device: {device}")

    # Create a simple test configuration
    with hydra.initialize(config_path="../../config", version_base="1.3"):
        logging.info("Loading configuration...")
        # Use test config
        cfg = hydra.compose(config_name="train")

        # Instantiate the model
        t2 = hydra.utils.instantiate(cfg.model)
        # dtype = torch.bfloat16
        dtype = torch.float32
        t2 = t2.to(device, dtype=dtype)
        # default
        # t2 = torch.compile(t2)

        # reduce-overhead
        # t2 = torch.compile(t2, mode="reduce-overhead")
        # max-autotune
        # t2 = torch.compile(t2, mode="max-autotune")

        # max-autotune-no-cudagraphs
        # t2 = torch.compile(t2, mode="max-autotune-no-cudagraphs")

        optimizer = hydra.utils.instantiate(cfg.optimizer, params=t2.parameters())

        lr_scheduler = hydra.utils.instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            num_training_steps=num_steps,
        )

        # Create random batch with config
        batch = create_random_batch(batch_size=1, device=device, dtype=dtype, cfg=cfg)
        batch = {
            k: v.repeat(num_repeat_batch, *[1] * (v.ndim - 1)) for k, v in batch.items()
        }
        for k, v in batch.items():
            if "mask" in k:
                assert v.dtype == torch.bool
            else:
                assert v.dtype == dtype

        task_names = list(cfg.model.tasks.keys())

        logging.info("Running forward pass...")
        t2.train()
        for step_idx in range(num_steps):
            # Test forward pass
            losses = t2([(task_names, batch)], is_eval=False)
            optimizer.zero_grad()
            losses["loss"].squeeze().backward()
            optimizer.step()
            lr_scheduler.step()

            for name, loss in losses.items():
                if type(loss) is float:
                    continue

                # Check loss is a scalar tensor
                assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
                assert loss.shape == torch.Size([]), (
                    f"Loss should be a scalar, got shape {loss.shape}"
                )
                assert not torch.isnan(loss), "Loss is NaN"
                assert not torch.isinf(loss), "Loss is Infinity"

            if step_idx % 100 == 0:
                t2.eval()
                with torch.inference_mode():
                    losses = t2([(task_names, batch)], is_eval=True)
                    for name, loss in losses.items():
                        if "ctrl/" not in name:
                            continue
                        if type(loss) is float:
                            continue
                        logging.info(f"eval/{name}: {loss.item()}")
                t2.train()

        logging.info("Test completed successfully!")
        return loss


if __name__ == "__main__":
    set_torch_configs()
    logging.basicConfig(level=logging.INFO)
    test_t2_forward()
