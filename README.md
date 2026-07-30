<h1> Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design</h1>
<div style="text-align: center;">

[Huy Ha](https://huy-ha.github.io/)$^{1,2}$, [C. Karen Liu](https://tml.stanford.edu)$^{1}$, [Shuran Song](https://shurans.github.io/)$^{1,2}$

$^1$ Stanford University, $^2$ Columbia University

[Project Page](https://transformer-transformer.github.io) | [Paper](https://arxiv.org/abs/2607.25798) | [Video](https://youtu.be/TTyjvPVFbNw)

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="docs/assets/looping_robotokens.gif">

Not all robots are created equal — but what if you could design one for the task at hand?
Transformer Transformer is a unified model that does exactly this: hand it a manipulation demonstration, and it generates a complete robot — every link, joint, motor, and inertial property — optimized for that motion.

</div>
</div>

<br>

We fabricated one of its designs for cloth flinging on a bimanual ALOHA platform — it cut tracking error by 73% and peak joint speed by 30% compared to the original ALOHA ([see it in action](https://youtu.be/TTyjvPVFbNw)).

This repository contains the full stack behind those results: RoboTokens (a unified tokenization of robot embodiments, states, and actions from MuJoCo models), procedural robot generation over parameterized design spaces, data generation with Mink and RL controllers, Transformer Transformer training and inference for cross-embodiment control and motion-conditioned robot co-design, CMA-ES baselines, and the Blender pipelines behind all our robot visualizations.

If you have any questions, please contact [Huy Ha](https://huy-ha.github.io/) at `huyha [at] stanford [dot] edu`.

**Table of Contents**

If you just want to start running some commands while skimming the paper, you should [get started here](docs/starter.md), which downloads checkpoints and data, then evaluates a pretrained Transformer Transformer on control and co-design.

- 🏃‍♀️ [Getting Started](docs/starter.md)
  - ⚙️ [Setup](docs/starter.md#setup)
  - 📍 [Checkpoints & Data](docs/starter.md#checkpoints--data)
  - 📊 [Evaluation](docs/starter.md#evaluation)
- 🗄️ [Data Generation](docs/data_generation.md)
  - 🤖 [RoboTokens](docs/data_generation.md#robotokens)
  - 🕹️ [Controllers: Mink v.s. RL](docs/data_generation.md#controllers-mink-vs-rl)
  - 🧬 [CMA-ES](docs/data_generation.md#cma-es)
  - ⚡ [Parallelization](docs/data_generation.md#parallelization)
  - 📱 [UMI Data Collection & Processing](docs/data_generation.md#umi-data-collection--processing)
  - 💾 [Data Format](docs/data_generation.md#data-format)
- 🚂 [Model Training](docs/training.md)
  - 🎛️ [Hydra Configs](docs/training.md#hydra-configs)
  - 🎮 [Control Only](docs/training.md#control-only)
  - 🦾 [Hardware Generation Only](docs/training.md#hardware-generation-only)
  - 🧠 [Control + Hardware Generation](docs/training.md#control--hardware-generation)
- 🔭 [Extending](docs/extending.md)
  - 🌱 [Adding New Design Spaces](docs/extending.md#adding-new-design-spaces)
  - 🧩 [Adding New RoboToken Fields](docs/extending.md#adding-new-robotoken-fields)
  - 🚧 [Limitations](docs/extending.md#limitations)
- 📽️ [Visualizations](docs/visualization.md)
  - 🎬 [Robot Rollouts](docs/visualization.md#robot-rollouts)
  - 🌪️ [Diffusion Processes](docs/visualization.md#diffusion-processes)

# Citation

If you find this work useful, please consider citing:

```bibtex
@article{ha2026transformer,
  title={Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design},
  author={Ha, Huy and Liu, C. Karen and Song, Shuran},
  journal={arXiv preprint arXiv:2607.25798},
  year={2026}
}
```

# Code Acknowledgements

**Model**:

- The Transformer Transformer backbone is modified from Meta's [DiT](https://github.com/facebookresearch/DiT), which also carries code from [GLIDE](https://github.com/openai/glide-text2im) and [MAE](https://github.com/facebookresearch/mae). These derived files (`t2/model/core.py` and the positional embedding utilities in `t2/model/utils.py`) remain under their upstream Attribution-NonCommercial licenses; the rest of this repository is MIT licensed.
- Rotation conversion utilities in `t2/train/augment.py` are copied from [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (BSD).

**Simulation & Control**:

- Physics runs on [MuJoCo and MuJoCo MJX](https://github.com/google-deepmind/mujoco) from Google DeepMind.
- Our MJX environments and RL training loop build on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and [Brax](https://github.com/google/brax)'s PPO implementation, with data augmentations adapted from [JaxRL](https://github.com/ikostrikov/jaxrl).
- The differential-IK oracle controllers are built on [mink](https://github.com/kevinzakka/mink). Big shout out to [Kevin Zakka](https://kzakka.com/) for his consistently excellent open-source robotics work — go give him a few stars ⭐
- The CMA-ES baseline uses [pycma](https://github.com/CMA-ES/pycma) by Nikolaus Hansen, and [evosax](https://github.com/RobertTLange/evosax) powers evolution strategies on GPU.

**Robot Models**:

- Robot assets under `assets/mjcf/` are modified from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — ALOHA 2 (Trossen Robotics), Agility Cassie, ANYmal B & C (ANYbotics), Unitree A1/Go1/Go2/H1, UR5e & UR10e (Universal Robots), and the Wonik Allegro hand. Each directory keeps its original license file.
- The quadruped manipulator assets combine the Unitree Go2 with the ARX5 arm, plus the Fin-Ray gripper and GoPro mount from [UMI](https://umi-gripper.github.io/), assembled originally for [UMI on Legs](https://umi-on-legs.github.io/).

**UMI Data**:

- Human demonstrations were collected with the [Universal Manipulation Interface](https://github.com/real-stanford/universal_manipulation_interface) and [iPhUMI](https://github.com/real-stanford/iPhUMI), UMI's iPhone-tracked extension.