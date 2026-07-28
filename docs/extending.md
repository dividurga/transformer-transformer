# Extending

This page covers the two most common ways to build on this codebase.
Adding a new design space requires no Python changes to the tokenizer or the model — it is mostly MJCF authoring.
Adding a new RoboToken field touches a well-defined chain of files, and I'll walk through every link in that chain.

## Adding New Design Spaces

A design space in this codebase is defined entirely inside MJCF files.
There are no per-robot sampling rules in Python.
Instead, each XML declares its own variation using MuJoCo's `<custom>` block, and a small generic parser in [`t2/robogen/components.py`](../t2/robogen/components.py) turns a seed (or an explicit parameter vector) into a concrete robot.

### Continuous and discrete parameters with `<numeric>`

Each `<numeric>` element's `name` is a backslash-delimited 6-tuple, and its `data` holds the sampling support.

```
identifier\namespace\attr_name\idx\randomization_type\value_resolver_lambda
```

- `identifier` — the MJCF element name to edit, e.g. `base_link` or `left:shoulder_mount`. Any `@` in the identifier is replaced with dm_control's `PREFIX_SEPARATOR`, so you can target elements inside attached components.
- `namespace` — the element kind, e.g. `body`, `site`, `geom`.
- `attr_name` — the attribute to mutate, e.g. `pos`, `euler`, `size`.
- `idx` — index into the attribute array (`0`/`1`/`2`), or `:` for the whole vector.
- `randomization_type` — `range` samples uniformly between the two `data` values; `choice` picks one row of `data`.
- `value_resolver_lambda` — a tiny Python expression evaluated with `x` bound to the attribute's original value and `y` bound to the sampled value.

Here is a real excerpt from [`assets/mjcf/aloha/aloha_base_bimanual_v2.xml`](../assets/mjcf/aloha/aloha_base_bimanual_v2.xml), the V2 side-by-side bimanual ViperX base.

```xml
<custom>
    <numeric name="base_link\body\pos\0\range\y" size='2' data='-0.8 -0.1' />
    <numeric name="base_link\body\pos\1\range\y" size='2' data='-0.0 0.0' />
    <numeric name="base_link\body\pos\2\range\y" size='2' data='-0.1 0.1' />

    <numeric name="left:shoulder_mount\site\pos\1\range\y" size='2' data='0.20 0.45' />
    <text name="symmetric_arm_spacing"
        data="left:shoulder_mount\site\pos\1\range\y;right:shoulder_mount\site\pos\1\range\-y;left_base_collision\geom\pos\1\range\y;left_base_visual\geom\pos\1\range\y;right_base_collision\geom\pos\1\range\-y;right_base_visual\geom\pos\1\range\-y" />

    <numeric name="left:shoulder_mount\site\euler\2\range\y" size='2'
        data='-3.141592653589793 3.141592653589793' />
    <text name="symmetric_arm_yaw"
        data="left:shoulder_mount\site\euler\2\range\y;right:shoulder_mount\site\euler\2\range\-y" />
</custom>
```

The first three lines vary the base placement.
The fourth samples the left arm's lateral mount offset along y in [0.20, 0.45] — half the distance between the two arms.

### Coupled edits with `<text>` constraint groups

A `<text>` element's `data` is a `;`-separated list of `<numeric>` specs that must all resolve from the *same* sampled value.
This is how symmetric left/right structure stays symmetric: `symmetric_arm_spacing` above samples one `y`, then applies it to six elements.
Each member still applies its own resolver lambda, which is how the right side mirrors with `-y` while the left side takes `y` directly.

Resolver lambdas get more interesting when one sampled scalar has to drive several linked geometry edits.
From [`assets/mjcf/aloha/aloha_base_upside_down_bimanual_50cm_xz_length_variation.xml`](../assets/mjcf/aloha/aloha_base_upside_down_bimanual_50cm_xz_length_variation.xml), a single arm-length sample `y` stretches the extension geom with `x+y` (offset the original half-size), shifts the connector with `(y-x)*2`, and re-places the child link with `0.3+y*2` — all in one constraint group.

> 📘 **Info**
>
> The resolver lambda is `eval`'d Python with only `x` (original value) and `y` (sampled value) in scope.
> Keep them to one-line arithmetic.
> If you find yourself wanting a real function, you probably want two `<numeric>` fields in one `<text>` group instead.

### Discrete topology with `*_mount` sites

Component attachment points are `<site>` elements whose name ends in `_mount`.
The part before `_mount` (after the last `:`) is a `|`-separated list of component names; if there is more than one option, a discrete choice is consumed.
From the quadruped manipulator design space (`assets/mjcf/umi_on_legs_plus_plus/quadruped_manipulator.xml`)

```xml
<site name="FL:left_leg|left_linkage_leg_mount" ... />
<site name="arx5|rail_mount" ... />
```

the front-left hip picks between a standard leg and a linkage leg, and the torso picks between mounting the ARX5 arm directly or on a rail.
A single-option site like `name="left:shoulder_mount"` always attaches `shoulder` and consumes no choice.
Attachment recurses: mounted components can have `*_mount` sites of their own.

### The Python side

All of the parsing lives in [`t2/robogen/components.py`](../t2/robogen/components.py) and is shared by every design space.

- `grow_components(seed, components, ...)` — the seed-driven builder. Loads the base XML, BFS-traverses `*_mount` sites attaching children, and calls `resolve_variations` to sample every `<numeric>` (honoring `<text>` groups) with a `np.random.RandomState(seed)`.
- `grow_components_with_params(components, choices, uniforms, ...)` — the deterministic twin. Consumes explicit integer `choices` for each `|` branch and `uniforms` in [0, 1] for each `range` field. This flat vector is exactly what CMA-ES optimizes over.
- `enumerate_choices(components, ...)` — enumerates the full discrete design space, returning `(choices, num_uniforms)` pairs.
- `make_robot(components, ...)` — binds a components dict to picklable partials `(seed_fn, from_params_fn, from_choices_fn)`.

A design space module is then just a dict and one call.
From [`t2/robogen/viperx.py`](../t2/robogen/viperx.py)

```python
VIPERX_BIMANUAL_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_bimanual.xml",
    **BIMANUAL_ARM_COMPONENTS,  # shoulder, upper_arm, ..., gripper XMLs
}
(
    viperx_bimanual_full_variation,
    viperx_bimanual_full_variation_from_params,
    _,
) = make_robot(
    VIPERX_BIMANUAL_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)
```

`resolve_before_mount=True` resolves each component's own `<custom>` fields as it is loaded, before dm_control prefixes its names by attaching it.
Leave it at the default `False` if your base XML's fields need to reference elements inside already-attached children.

### Wiring into configs

Data generation and evaluation instantiate the seed function through Hydra, from `config/env/*.yaml`

```yaml
robot_generator:
  _target_: t2.robogen.viperx.viperx_bimanual_full_variation
  _partial_: true
```

while the CMA-ES baseline needs the components dict (to enumerate choices) plus the params function, from `config/optimize_cmaes_viperx_bimanual.yaml`

```yaml
choice_components: t2.robogen.viperx.VIPERX_BIMANUAL_COMPONENTS
robogen_from_params_fn:
  _target_: t2.robogen.viperx.viperx_bimanual_full_variation_from_params
```

Envs that pin a specific point in the design space pass `choices`/`uniforms` explicitly (see `config/env/umi_on_legs_plus_plus_from_params.yaml` and `config/generate_robogen_tokens.yaml`).

### Worked example: the bimanual ViperX variants

Our real-world ALOHA bimanual space is a catalogue of variants that all share the same arm component XMLs (`BIMANUAL_ARM_COMPONENTS`) and differ *only* in the base XML, which carries all the `<custom>` variation fields.
I like this one-base-XML-per-variant pattern: each variant is diffable in a text editor, and growing a family (V0 → V1 → V2) means copying the base XML and adding one constraint group.

| Variant         | Function in `t2/robogen/viperx.py` | Base XML in `assets/mjcf/aloha/`         | Variation channels                        |
| --------------- | ---------------------------------- | ---------------------------------------- | ----------------------------------------- |
| Side-by-side V0 | `viperx_bimanual_full_variation`   | `aloha_base_bimanual.xml`                | base x/y/z placement                      |
| Side-by-side V1 | `viperx_bimanual_v1`               | `aloha_base_bimanual_v1.xml`             | + symmetric arm spacing (0.40–0.90 m)     |
| Side-by-side V2 | `viperx_bimanual_v2`               | `aloha_base_bimanual_v2.xml`             | + symmetric mount yaw (left +θ, right −θ) |
| Opposing V0     | `viperx_bimanual_opposing`         | `aloha_base_bimanual_opposing.xml`       | base x/y/z placement                      |
| Opposing V1     | `viperx_bimanual_opposing_v1`      | `aloha_base_bimanual_opposing_v1.xml`    | + symmetric arm spacing                   |
| Upside-down V0  | `viperx_upside_down_bimanual`      | `aloha_base_upside_down_bimanual.xml`    | base x/y/z + symmetric mount x/z          |
| Upside-down V1  | `viperx_upside_down_bimanual_v1`   | `aloha_base_upside_down_bimanual_v1.xml` | + symmetric arm spacing                   |
| Upside-down V2  | `viperx_upside_down_bimanual_v2`   | `aloha_base_upside_down_bimanual_v2.xml` | + symmetric mount yaw                     |
| Vertical V0     | `viperx_vertical_bimanual`         | `aloha_base_vertical_bimanual.xml`       | base x/y/z + symmetric mount x/z          |
| Vertical V1     | `viperx_vertical_bimanual_v1`      | `aloha_base_vertical_bimanual_v1.xml`    | + symmetric arm spacing                   |
| Vertical V2     | `viperx_vertical_bimanual_v2`      | `aloha_base_vertical_bimanual_v2.xml`    | + symmetric mount yaw                     |

Two functions skip the DSL entirely and just load a single fixed XML: `viperx_bimanual_opposing_70cm` and `viperx_upside_down_bimanual_50cm`.
That is the pattern to copy when you need one exact robot, e.g. to match physical hardware.
The `viperx_upside_down_bimanual_50cm_xz_variation` / `_xz_length_variation` / `_xyz_length_variation` families then re-open variation channels around that fixed real-world robot — mount placement first, then arm-link lengths — for hardware generation (`hardware_gen`) experiments anchored to the real ALOHA cell.

### Recipe: a new design space end-to-end

1. Author your MJCF components under `assets/mjcf/<name>/`: one base XML plus one XML per mountable component. Collision geoms must be primitives (sphere, capsule, cylinder, box).
2. Add a `<custom>` block with `<numeric>` fields for every varying parameter, `<text>` groups for coupled/symmetric edits, and `*_mount` sites (with `|` options) wherever the topology should branch.
3. Create `t2/robogen/<name>.py` with a `<NAME>_COMPONENTS` dict and a `make_robot(...)` call.
4. Add `config/env/<name>.yaml` pointing `robot_generator._target_` at the seed function with `_partial_: true`, and select it from `config/datagen.yaml` to generate rollouts.
5. If you want the CMA-ES baseline, add a `config/optimize_cmaes_<name>.yaml` with `choice_components` and `robogen_from_params_fn`.
6. Before training, add a `config/addon_*` config with your design space's per-modality token counts (`seq_len.dyna_joint`, `seq_len.link`, ...) — [`config/addon_hardware_viperx.yaml`](../config/addon_hardware_viperx.yaml) is the template. Count tokens off a tokenized worst-case robot from your space.

Tokenization is generic — any primitive-geometry MJCF works, so nothing in `t2/robotok/` needs changing.

Sanity check the discrete structure of your space with the enumeration test pattern.

```sh
pytest tests/robogen/test_enumeration.py -v
```

This enumerates every `(choices, num_uniforms)` pair of the quadruped manipulator space, builds each design, and asserts that over- and under-specified parameter vectors raise.
Point it at your own `COMPONENTS` dict by copying the test.

> 🪲 **Troubleshooting silent variations**
>
> `resolve_variations` skips `<numeric>` fields whose `identifier` can't be found (`STRICT_RESOLUTION = False` in `t2/robogen/components.py`).
> If a variation silently never applies, the identifier is almost certainly misspelled or missing its `left:`-style prefix.
> I flip `STRICT_RESOLUTION` to `True` while authoring a new space — it turns typos into loud `Element not found` errors.

> 🪲 **Troubleshooting parameter counts**
>
> `grow_components_with_params` raises `ValueError: uniforms overspecified` (or `choices overspecified`) when you pass too many parameters, and `IndexError` when you pass too few.
> Get the exact counts for each branch from `enumerate_choices(COMPONENTS)` instead of counting by hand.

## Adding New RoboToken Fields

A RoboToken field lives in a chain: dataclass → tokenizer → serialization schema → zarr writer → dataset loader → model adapter config.
The middle of the chain is generic and mostly leaves you alone.
Here is the full recipe, using a scalar on `DynamicJoint` as the running example.

1. **Dataclass** — [`t2/robotok/token.py`](../t2/robotok/token.py).
Add the field to the frozen pydantic dataclass (`Link`, `DynamicJoint`, `Actuator`, ...).
Discrete fields should follow the enum pattern (`GeomType`, `DynamicJointType`, `ActuatorType`) with `to_int`/`from_int`.

2. **Tokenizer** — [`t2/robotok/tokenizer.py`](../t2/robotok/tokenizer.py).
Read the value from the compiled `MjModel` inside `tokenize()` and pass it through the relevant helper: joint fields flow through `_tokenize_joint` (see how `armature` is read from `model.dof_armature` and `spring_ref` from `model.qpos_spring`), link fields through `_tokenize_body`, actuator fields through `_tokenize_actuators`.
Then emit it back to MJCF in `detokenize()` — if you skip this, the round-trip tests will catch you.

3. **Serialization** — [`t2/robotok/io.py`](../t2/robotok/io.py).
Register the key in `get_robot_serialization_schema` (e.g. `"dyna_joint/my_field": []`), write it in `serialize()`, and read it back in `deserialize()`.
Note that every joint emits **two** tokens, one per `JointLinkConnection`, so joint-level values are duplicated across both connections (`.extend([[value]] * 2)`) while per-connection values like `pos` and `rotmat` differ.
Link and actuator fields are appended once per token.

4. **Zarr writer** — [`t2/io/schema.py`](../t2/io/schema.py).
Nothing to do for a hardware field: `append_hardware` groups keys by their prefix, and any prefix in `link`/`dyna_joint`/`fixed_joint`/`actuator` flows through automatically.
A **new rollout field** is the one case that needs a code change — add its group prefix to `ALLOWED_ROLLOUT_GROUPS`.

5. **Dataset loader** — [`t2/data/dataset.py`](../t2/data/dataset.py).
Nothing to do, but know the invariant: the writer concatenates all attrs of a group along the last axis in **alphabetical order** (`sorted_dict`) and records each attr's width in the zarr array's `attrs`; the loader iterates the same `sorted_dict` and slices widths back out.
Both sides sort, so the layout is stable — never rely on insertion order, and never reorder one side without the other.

6. **Adapter config** — `config/model/tasks/adapter/<group>.yaml`.
This is the required manual step: add `my_field: <width>` under `modality.attrs` in the group's base adapter (e.g. [`dyna_joint.yaml`](../config/model/tasks/adapter/dyna_joint.yaml)); the `*_diffuse.yaml` variants inherit it.
The width must equal the last-axis width you serialized.
Discrete fields additionally need `attr_encoders`/`attr_decoders` entries (`t2.model.encoding.BinaryEncoding`/`BinaryDecoding`) with the width set to a `dims.*_bits` entry you add in [`config/train.yaml`](../config/train.yaml).

Finally, regenerate your datasets — attr widths are baked into the zarr arrays at write time, so old data does not contain the new field.

> ❗**Caution**
>
> Old checkpoints will not load cleanly after changing `modality.attrs`, since adapter input/output projections change shape.
> Treat a schema change as a new dataset + new training run.

### Tokenizer conventions worth knowing

- Transforms are named `T_ab`: the pose of frame `b` expressed in frame `a`. Quaternions are `wxyz` order, angles in radians.
- The tokenized robot is canonical: every link has exactly one collision geom centered at its origin, and every joint's axis is `(0, 0, 1)` in its own local frame. `JointLinkConnection.pose` is the link geom's pose in the joint's frame.
- `preprocess_mjcf` produces that canonical form before tokenization: it normalizes geom transforms, converts tracking sites, adds collision boxes where needed, and splits bodies with multiple collision geoms into one body per primitive.
- When a multi-geom body is split, mass is divided per-primitive under a uniform-density assumption (`density = body_mass / total_primitive_volume`), and the body's inertia is redistributed: satellite geoms get analytic primitive inertias shifted to the body's inertia frame, and the largest geom keeps the remainder so the totals match MuJoCo's original body exactly.

### Verifying with the round-trip tests

Run the RoboToken test suite from the repo root.

```sh
pytest tests/robotok -v
```

These tests build randomized MJCF models, then check `preprocess → tokenize → detokenize` *and* `tokenize → serialize → deserialize → detokenize` against the preprocessed original with `check_physics_equal` — mass matrices, inertias, and stepped states must match.
If your new field affects physics and you forgot step 2's `detokenize()` or step 3's `deserialize()`, these tests fail.

To add a round-trip case for a new field, copy a case in `tests/robotok/test_core.py`: build a minimal model with `default_root_element()`, set your new attribute to a randomized non-default value (parameterized over seeds, so the default value can't mask a dropped field), and call `tokenize_detokenize(root)` from `tests/robotok/utils.py`.

```python
@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_my_field(seed: int):
    root = default_root_element()
    rs = np.random.RandomState(seed)
    body = root.worldbody.add("body", name="link_0", pos=rs.uniform(-0.1, 0.1, 3))
    body.add("geom", name="geom_0", **TEST_LINK_GEOM_ATTRS)
    body.add("joint", name="joint_0", **{**TEST_JOINT_ATTRS, "my_field": rs.uniform(0.0, 1.0)})
    tokenize_detokenize(root)
```

If the field survives serialization and the rebuilt physics still matches, you are done — every downstream consumer from the zarr writer to the model adapters will pick it up from there.

## Limitations

RoboTokens represent geometry with primitives (spheres, boxes, cylinders, capsules) — the model does not generate complex 3D geometry like meshes or TSDFs.

Here is the thing though: adding a complex 3D geometry representation (vertices + edges + faces, TSDF, etc.) isn't the hard part.
The hard part is the data generation problem — getting meaningfully *diverse* robot geometry across *complete* robot embodiments.
If the model doesn't have very diverse complex 3D geometry data to train on, its 3D geometry generation will collapse to one-in-k predictions, and you'll have spent a lot of the model's compute on an expressive representation of a not-very-expressive design space.
This difficulty is one of the reasons why complex geometry was left for future work.