import typing

import hydra
import torch
import torch.nn as nn
import transformers
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf

from t2.model.modality import DiffusionModalityAdapter
from t2.model.t2 import T2
from t2.model.task.task import MultiModalDiffusionTask
from t2.train.utils import set_torch_configs
from t2.utils.misc import seed_everything

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))


def test_overfit(
    num_steps: int = 100000, batch_size: int = 1, num_repeat_batch: int = 256
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(0)

    # Create a simple test configuration
    with hydra.initialize(config_path="config", version_base="1.3"):
        print("Loading configuration...")
        cfg = hydra.compose(config_name="overfit_model")

        # Instantiate the model
        model: T2 = hydra.utils.instantiate(cfg.model)
        # dtype = torch.bfloat16
        dtype = torch.float32
        model = model.to(device, dtype=dtype)
        # default
        # model = torch.compile(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        lr_scheduler = transformers.get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_steps,
        )

        # Create random batch with config
        batch = {
            "test_modality/test_attr_1": torch.rand(
                batch_size, cfg.seq_len.test_seq_len, 1, device=device
            )
            * 2
            - 1,
            "test_modality/test_attr_2": torch.rand(
                batch_size, cfg.seq_len.test_seq_len, 1, device=device
            )
            * 2
            - 1,
            "test_modality/mask": torch.zeros(
                batch_size,
                cfg.seq_len.test_seq_len,
                1,
                device=device,
                dtype=torch.bool,
            ),
            "test_modality/test_pos_idx/id": torch.arange(
                0,
                cfg.seq_len.test_seq_len,
                device=device,
            )[None, :].repeat(batch_size, 1),
        }
        batch = {
            k: v.repeat(num_repeat_batch, *[1] * (v.ndim - 1)) for k, v in batch.items()
        }
        print("Running forward pass...")
        model.train()
        ema = EMAModel(
            model.parameters(),
            power=0.75,
            update_after_step=0,
        )
        test_task = typing.cast(MultiModalDiffusionTask, model.tasks["test_task"])
        adapter = typing.cast(
            DiffusionModalityAdapter, test_task.adapters["test_adapter"]
        )
        for step_idx in range(num_steps):
            # Test forward pass
            losses = model([(["test_task"], batch)], is_eval=False)
            optimizer.zero_grad()
            losses["loss"].squeeze().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            ema.step(model.parameters())
            if step_idx % 500 == 0:
                print("=" * 50)
                print(f"Step {step_idx}")
                model.eval()
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                with torch.inference_mode():
                    # do entire diffusion process
                    samples = test_task.decode(
                        batch={k: v[:batch_size] for k, v in batch.items()},
                        backbone=model.backbone,
                        pos_embs=model.pos_embs,
                        num_inference_steps=test_task.noise_scheduler.config.num_train_timesteps,
                    )["test_modality/test_attr_1"]
                    err = nn.functional.mse_loss(
                        samples, batch["test_modality/test_attr_1"][:batch_size]
                    )
                    print(f"Err: {err.item():.08f}")
                    print(f"Loss: {losses['loss'].item():.08f}")
                    print(f"Samples: {samples[:batch_size].view(-1)}")
                    print(
                        f"Target: {batch['test_modality/test_attr_1'][:batch_size].view(-1)}"
                    )
                ema.restore(model.parameters())
                model.train()

        print("Test completed successfully!")
        return losses["loss"]


if __name__ == "__main__":
    set_torch_configs()
    test_overfit()
