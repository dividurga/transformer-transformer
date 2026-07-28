import copy

import torch
from dm_control import mjcf

from t2.robogen.viperx import viperx
from t2.robotok.io import deserialize, serialize
from t2.robotok.tokenizer import detokenize, preprocess_mjcf, tokenize
from t2.train.augment import TransformAugmentation


def test_transform_augmentation():
    augmentation = TransformAugmentation(
        pos_aug_magnitude=(0.1, 0.1, 0.1),
        orn_aug_magnitude=(0.0, 0.0, 0.5),
        affected_pose_fields=[],
    )
    mjcf_model = viperx(0)
    mjcf_model = preprocess_mjcf(mjcf_model)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)
    assert physics is not None
    for _ in range(10):
        physics.step()
    tokens, _, _, _ = tokenize(mjcf_model)
    robot_dict = serialize(tokens)
    batch = {k: torch.tensor(v)[None, ...].float() for k, v in robot_dict.items()}
    group_keys: dict[str, str] = {}
    for k in robot_dict:
        group_keys.setdefault(k.split("/")[0], k)
    for group, key in group_keys.items():
        seq_len = batch[key].shape[1]
        batch[f"{group}/mask"] = torch.zeros(1, seq_len, 1, dtype=torch.bool)
    out = augmentation(copy.deepcopy(batch))
    out = {k: v for k, v in out.items() if not k.endswith("/mask")}
    new_robot_dict = {k: v.squeeze(0).numpy() for k, v in out.items()}
    new_tokens = deserialize(new_robot_dict)
    new_mjcf, _, _, _ = detokenize(
        new_tokens, gravcomp=True
    )  # true for only fixed base robots
    new_physics = mjcf.Physics.from_mjcf_model(new_mjcf)
    assert new_physics is not None
    for _ in range(10):
        new_physics.step()


if __name__ == "__main__":
    test_transform_augmentation()
