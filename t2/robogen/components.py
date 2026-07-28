import copy
import functools
import os

import numpy as np
from dm_control import mjcf
from dm_control.mjcf.constants import PREFIX_SEPARATOR

STRICT_RESOLUTION = False


def enable_gravity_compensation(
    mjcf_model: mjcf.RootElement,
) -> mjcf.RootElement:
    for body in mjcf_model.find_all("body"):
        children = body.all_children()
        if len(children) == 1 and children[0].tag == "body":
            continue
        body.gravcomp = 1
    return mjcf_model


def resolve_variation(
    rs: np.random.RandomState,
    root: mjcf.RootElement,
    namespace: str,
    identifier: str,
    attr_name: str,
    randomization_type: str,
    value_resolver_lambda: str,
    attr_data_idx: int,
    randomization_data: np.ndarray,
    randomized_value: float | int | None = None,
):
    element = root.find(
        namespace=namespace,
        identifier=identifier,
    )
    if element is None:
        raise ValueError(
            f"Element not found: {namespace} {identifier} {attr_name} {attr_data_idx}"
        )
    attr = getattr(element, attr_name)
    attr_is_array = isinstance(attr, np.ndarray)
    if attr_is_array:
        if attr_data_idx != ":":
            original_value = attr[int(attr_data_idx)]
        else:
            original_value = attr[:]
    else:
        original_value = attr
    if randomized_value is None:
        if attr_is_array:
            randomization_data = randomization_data.reshape(-1, *original_value.shape)
        if randomization_type == "range":
            assert len(randomization_data) == 2
            randomized_value = rs.uniform(randomization_data[0], randomization_data[1])
        elif randomization_type == "choice":
            choice_idx = rs.choice(len(randomization_data))
            randomized_value = randomization_data[choice_idx]
        else:
            raise ValueError(f"Unknown randomization type: {randomization_type}")

    new_value = eval(
        value_resolver_lambda, {"x": original_value, "y": randomized_value}
    )
    if attr_is_array:
        if attr_data_idx != ":":
            attr[int(attr_data_idx)] = new_value
        else:
            attr[:] = new_value
    else:
        setattr(element, attr_name, new_value)

    return randomized_value


def resolve_variations(
    rs: np.random.RandomState,
    root: mjcf.RootElement,
) -> mjcf.RootElement:
    variation_constraints = {}

    for variation_constraint in root.find_all("text"):
        variation_constraints[variation_constraint.name] = set(
            variation_constraint.data.split(";"),
        )

    resolved_variations = set()

    for v in root.find_all("numeric"):
        field_name = v.name
        prefix = "/".join(v.full_identifier.split("/")[:-1])
        if field_name in resolved_variations:
            continue
        (
            identifier,
            namespace,
            attr_name,
            idx,
            randomization_type,
            value_resolver_lambda,
        ) = field_name.split("\\")
        try:
            randomized_value = resolve_variation(
                rs=rs,
                root=root,
                namespace=namespace,
                identifier=os.path.join(prefix, identifier).replace(
                    "@", PREFIX_SEPARATOR
                ),
                attr_name=attr_name,
                randomization_type=randomization_type,
                value_resolver_lambda=value_resolver_lambda,
                attr_data_idx=idx,
                randomization_data=v.data,
            )
        except ValueError as e:
            if STRICT_RESOLUTION:
                raise e
            continue
        resolved_variations.add(v.name)
        for constraint_group in variation_constraints.values():
            if field_name in constraint_group:
                for other_field_name in constraint_group:
                    if other_field_name == field_name:
                        continue
                    assert other_field_name not in resolved_variations
                    (
                        identifier,
                        namespace,
                        attr_name,
                        idx,
                        randomization_type,
                        value_resolver_lambda,
                    ) = other_field_name.split("\\")
                    resolve_variation(
                        rs=rs,
                        root=root,
                        namespace=namespace,
                        identifier=os.path.join(prefix, identifier).replace(
                            "@", PREFIX_SEPARATOR
                        ),
                        attr_name=attr_name,
                        randomization_type=randomization_type,
                        value_resolver_lambda=value_resolver_lambda,
                        attr_data_idx=idx,
                        randomization_data=v.data,
                        randomized_value=randomized_value,
                    )
                    resolved_variations.add(other_field_name)
    return root


def resolve_actuators(
    component: mjcf.RootElement,
) -> mjcf.RootElement:
    # NOTE: right now actuator tuples are under utilized
    for tuple_element in component.find_all("tuple"):
        for actuator_info in tuple_element.all_children():
            component.actuator.add(
                "position",
                name=actuator_info.objname.name,
                joint=actuator_info.objname.name,
                dclass=actuator_info.objname.name,
            )
    return component


def grow_components(
    seed: int,
    components: dict[str, str],
    start_component: str = "base",
    gravity_compensation: bool = False,
    resolve_before_mount: bool = False,
) -> mjcf.RootElement:
    rs = np.random.RandomState(seed)
    base_path = components[start_component]
    keyframes = {}
    base = mjcf.from_path(base_path)
    for keyframe in base.find_all("key"):
        keyframes[keyframe.name] = {
            "qpos": keyframe.qpos,
            "ctrl": keyframe.ctrl,
        }
        keyframe.remove()
    if resolve_before_mount:
        base = resolve_variations(rs, base)
    unresolved_component_mounts = [base]
    while len(unresolved_component_mounts) > 0:
        component = unresolved_component_mounts.pop(0)
        for mount_site in component.find_all("site"):
            if not mount_site.name.endswith("_mount"):
                continue
            component_options = mount_site.name.split(":")[-1]
            component_options = component_options.split("_mount")[0].split("|")
            component_option_idx = (
                rs.choice(len(component_options)) if len(component_options) > 1 else 0
            )  # avoid updating random state if possible
            component_name = component_options[component_option_idx]
            component_path = components[component_name]
            component = mjcf.from_path(component_path)
            component = resolve_actuators(component)
            if resolve_before_mount:
                component = resolve_variations(rs, component)
            for keyframe in component.find_all("key"):
                assert keyframe.name in keyframes
                if keyframe.qpos is not None:
                    keyframes[keyframe.name]["qpos"] = (
                        np.concatenate(
                            [keyframes[keyframe.name]["qpos"], keyframe.qpos]
                        )
                        if keyframes[keyframe.name]["qpos"] is not None
                        else keyframe.qpos
                    )
                if keyframe.ctrl is not None:
                    keyframes[keyframe.name]["ctrl"] = (
                        np.concatenate(
                            [keyframes[keyframe.name]["ctrl"], keyframe.ctrl]
                        )
                        if keyframes[keyframe.name]["ctrl"] is not None
                        else keyframe.ctrl
                    )
                keyframe.remove()
            mount_site.attach(component)
            unresolved_component_mounts.append(component)

    if not resolve_before_mount:
        base = resolve_variations(rs, base)
    for name, keyframe in keyframes.items():
        base.keyframe.add(
            "key",
            name=name,
            qpos=keyframe["qpos"],
            ctrl=keyframe["ctrl"],
        )
    if gravity_compensation:
        base = enable_gravity_compensation(base)
    return base


def resolve_variation_with_params(
    choices: list[int],
    uniforms: list[float],
    root: mjcf.RootElement,
    namespace: str,
    identifier: str,
    attr_name: str,
    randomization_type: str,
    value_resolver_lambda: str,
    attr_data_idx: int,
    randomization_data: np.ndarray,
    randomized_value: float | int | None = None,
):
    element = root.find(
        namespace=namespace,
        identifier=identifier,
    )
    if element is None:
        available_options = root.find_all(namespace=namespace)
        option_names = []
        for option in available_options:
            option_names.append(option.full_identifier)
        raise ValueError(
            f"Element not found: {namespace} {identifier} {attr_name} {attr_data_idx} {option_names}"
        )
    attr = getattr(element, attr_name)
    attr_is_array = isinstance(attr, np.ndarray)
    if attr_is_array:
        if attr_data_idx != ":":
            original_value = attr[int(attr_data_idx)]
        else:
            original_value = attr[:]
    else:
        original_value = attr
    if randomized_value is None:
        if attr_is_array:
            randomization_data = randomization_data.reshape(-1, *original_value.shape)
        if randomization_type == "range":
            assert len(randomization_data) == 2
            uniform_value = uniforms.pop(0)
            randomized_value = (
                uniform_value * (randomization_data[1] - randomization_data[0])
                + randomization_data[0]
            )
        elif randomization_type == "choice":
            if len(randomization_data) > 1:
                choice_idx = choices.pop(0)
                choice_idx = choice_idx % len(randomization_data)
            else:
                choice_idx = 0
            randomized_value = randomization_data[choice_idx]
        else:
            raise ValueError(f"Unknown randomization type: {randomization_type}")

    new_value = eval(
        value_resolver_lambda, {"x": original_value, "y": randomized_value}
    )

    if attr_is_array:
        if attr_data_idx != ":":
            attr[int(attr_data_idx)] = new_value
        else:
            attr[:] = new_value
    else:
        setattr(element, attr_name, new_value)

    return randomized_value


def resolve_variations_with_params(
    choices: list[int],
    uniforms: list[float],
    root: mjcf.RootElement,
) -> mjcf.RootElement:
    variation_constraints = {}

    for variation_constraint in root.find_all("text"):
        variation_constraints[variation_constraint.name] = set(
            variation_constraint.data.split(";"),
        )

    resolved_variations = set()

    for v in root.find_all("numeric"):
        field_name = v.name
        prefix = "/".join(v.full_identifier.split("/")[:-1])
        if field_name in resolved_variations:
            continue
        (
            identifier,
            namespace,
            attr_name,
            idx,
            randomization_type,
            value_resolver_lambda,
        ) = field_name.split("\\")
        try:
            randomized_value = resolve_variation_with_params(
                choices=choices,
                uniforms=uniforms,
                root=root,
                namespace=namespace,
                identifier=os.path.join(prefix, identifier).replace(
                    "@", PREFIX_SEPARATOR
                ),
                attr_name=attr_name,
                randomization_type=randomization_type,
                value_resolver_lambda=value_resolver_lambda,
                attr_data_idx=idx,
                randomization_data=v.data,
            )
        except ValueError as e:
            if STRICT_RESOLUTION:
                raise e
            # `enumerate_choices` counts every <numeric> field, but some
            # identifiers only exist for a subset of the discrete mounts
            # (e.g. an arm-placement field that targets the direct-mount
            # prefix). Consume the parameters the field would have used so
            # the enumerated parameter counts stay exact.
            if randomization_type == "range" and len(uniforms) > 0:
                uniforms.pop(0)
            elif randomization_type == "choice" and len(choices) > 0:
                choices.pop(0)
            continue
        resolved_variations.add(v.name)
        for constraint_group in variation_constraints.values():
            if field_name in constraint_group:
                for other_field_name in constraint_group:
                    if other_field_name == field_name:
                        continue
                    assert other_field_name not in resolved_variations
                    (
                        identifier,
                        namespace,
                        attr_name,
                        idx,
                        randomization_type,
                        value_resolver_lambda,
                    ) = other_field_name.split("\\")
                    resolve_variation_with_params(
                        choices=choices,
                        uniforms=uniforms,
                        root=root,
                        namespace=namespace,
                        identifier=os.path.join(prefix, identifier).replace(
                            "@", PREFIX_SEPARATOR
                        ),
                        attr_name=attr_name,
                        randomization_type=randomization_type,
                        value_resolver_lambda=value_resolver_lambda,
                        attr_data_idx=idx,
                        randomization_data=v.data,
                        randomized_value=randomized_value,
                    )
                    resolved_variations.add(other_field_name)
    return root


def grow_components_with_params(
    components: dict[str, str],
    choices: list[int],
    uniforms: list[float],
    start_component: str = "base",
    gravity_compensation: bool = False,
    resolve_before_mount: bool = False,
) -> mjcf.RootElement:
    choices = copy.copy(choices)
    uniforms = copy.copy(uniforms)
    base_path = components[start_component]
    keyframes = {}
    base = mjcf.from_path(base_path)
    for keyframe in base.find_all("key"):
        keyframes[keyframe.name] = {
            "qpos": keyframe.qpos,
            "ctrl": keyframe.ctrl,
        }
        keyframe.remove()
    if resolve_before_mount:
        base = resolve_variations_with_params(choices, uniforms, base)
    unresolved_component_mounts = [base]
    while len(unresolved_component_mounts) > 0:
        component = unresolved_component_mounts.pop(0)
        for mount_site in component.find_all("site"):
            if not mount_site.name.endswith("_mount"):
                continue
            component_options = mount_site.name.split(":")[-1]
            component_options = component_options.split("_mount")[0].split("|")
            if len(component_options) > 1:
                component_option_idx = choices.pop(0) % len(component_options)
            else:
                # avoid updating choices if possible
                component_option_idx = 0

            component_name = component_options[component_option_idx]
            component_path = components[component_name]
            component = mjcf.from_path(component_path)
            component = resolve_actuators(component)
            if resolve_before_mount:
                component = resolve_variations_with_params(choices, uniforms, component)
            for keyframe in component.find_all("key"):
                assert keyframe.name in keyframes
                if keyframe.qpos is not None:
                    keyframes[keyframe.name]["qpos"] = (
                        np.concatenate(
                            [keyframes[keyframe.name]["qpos"], keyframe.qpos]
                        )
                        if keyframes[keyframe.name]["qpos"] is not None
                        else keyframe.qpos
                    )
                if keyframe.ctrl is not None:
                    keyframes[keyframe.name]["ctrl"] = (
                        np.concatenate(
                            [keyframes[keyframe.name]["ctrl"], keyframe.ctrl]
                        )
                        if keyframes[keyframe.name]["ctrl"] is not None
                        else keyframe.ctrl
                    )
                keyframe.remove()
            mount_site.attach(component)
            unresolved_component_mounts.append(component)

    if not resolve_before_mount:
        base = resolve_variations_with_params(choices, uniforms, base)
    if len(choices) != 0:
        raise ValueError("parameters overspecified")
    for name, keyframe in keyframes.items():
        base.keyframe.add(
            "key",
            name=name,
            qpos=keyframe["qpos"],
            ctrl=keyframe["ctrl"],
        )
    if gravity_compensation:
        base = enable_gravity_compensation(base)
    if len(uniforms) != 0:
        raise ValueError("uniforms overspecified")
    if len(choices) != 0:
        raise ValueError("choices overspecified")
    return base


def recurse_choice(
    current_choices: list[int],
    current_uniforms: list[str],
    unresolved_choices: list[int],
) -> list[tuple[list[int], list[str]]]:
    results = []
    if len(unresolved_choices) == 0:
        return [(current_choices, current_uniforms)]
    else:
        num_choices = unresolved_choices.pop(0)
        for choice_idx in range(num_choices):
            results.extend(
                recurse_choice(
                    current_choices + [choice_idx],
                    current_uniforms,
                    copy.copy(unresolved_choices),
                )
            )
        return results


def recurse_component_resolve_before_mount(
    component_name: str,
    current_choices: list[int],
    current_uniforms: list[str],
    unresolved_mount_sites: list[mjcf.Element],
    unresolved_choices: list[int],
    components: dict[str, str],
) -> list[tuple[list[int], list[str]]]:
    component_path = components[component_name]
    component = mjcf.from_path(component_path)

    # Count continuous variations for this component
    for numeric in component.find_all("numeric"):
        if "\\choice\\" not in numeric.name:
            current_uniforms.append(numeric.name)
        else:
            (
                identifier,
                namespace,
                attr_name,
                idx,
                randomization_type,
                value_resolver_lambda,
            ) = numeric.name.split("\\")
            element = component.find(namespace=namespace, identifier=identifier)
            attr = getattr(element, attr_name)
            num_choices = len(numeric.data.reshape(-1, *attr.shape))
            unresolved_choices.append(num_choices)

    # Find all mount sites in this component
    unresolved_mount_sites += [
        site
        for site in component.find_all("site")
        if getattr(site, "name", None) and getattr(site, "name").endswith("_mount")
    ]

    results = []
    # Base case: no mount sites
    if len(unresolved_mount_sites) == 0:
        if len(unresolved_choices) == 0:
            return [(current_choices, current_uniforms)]
        else:
            # recurse
            num_choices = unresolved_choices.pop(0)
            for choice_idx in range(num_choices):
                results.extend(
                    recurse_choice(
                        current_choices + [choice_idx],
                        current_uniforms,
                        copy.copy(unresolved_choices),
                    )
                )
            return results

    mount_site = unresolved_mount_sites.pop(0)
    mount_name = getattr(mount_site, "name", None)
    if not isinstance(mount_name, str):
        raise ValueError("Mount site missing string name")
    component_options = mount_name.split(":")[-1].split("_mount")[0].split("|")

    # If there are multiple options, we need to branch
    if len(component_options) > 1:
        for option_idx, option_name in enumerate(component_options):
            new_choices = current_choices + [option_idx]
            results.extend(
                recurse_component_resolve_before_mount(
                    option_name,
                    copy.copy(new_choices),
                    copy.copy(current_uniforms),
                    copy.copy(unresolved_mount_sites),
                    copy.copy(unresolved_choices),
                    components,
                )
            )
    else:
        # Single option, no choice needed
        option_name = component_options[0]
        results.extend(
            recurse_component_resolve_before_mount(
                option_name,
                copy.copy(current_choices),
                copy.copy(current_uniforms),
                copy.copy(unresolved_mount_sites),
                copy.copy(unresolved_choices),
                components,
            )
        )

    if len(results) == 0:
        return [(current_choices, current_uniforms)]

    return results


def recurse_component_resolve_after_mount(
    component_name: str,
    current_choices: list[int],
    current_uniforms: list[str],
    unresolved_mount_sites: list[mjcf.Element],
    unresolved_choices: list[int],
    components: dict[str, str],
    _root_component_name: str | None = None,
) -> list[tuple[list[int], list[str]]]:
    # NOTE: For resolve-before-mount, we can count variations per-component as we
    # traverse the component graph. For resolve-after-mount, variation resolution
    # happens once on the fully-mounted model, and constraint groups can couple
    # variations across attached subtrees. So we:
    # - enumerate mount-site branches first (BFS order matches grow_components),
    # - build the full model for each mount-choice prefix,
    # - then enumerate remaining choice-variations on the final model while
    #   respecting constraint groups (to avoid overspecifying `choices`).

    if _root_component_name is None:
        _root_component_name = component_name

    component_path = components[component_name]
    component = mjcf.from_path(component_path)

    # Find all mount sites in this component (BFS across the component tree).
    unresolved_mount_sites += [
        site
        for site in component.find_all("site")
        if getattr(site, "name", None) and getattr(site, "name").endswith("_mount")
    ]

    results: list[tuple[list[int], list[str]]] = []

    def build_fully_mounted_model(
        root_component_name: str,
        mount_choices: list[int],
    ) -> mjcf.RootElement:
        """Build full MJCF tree consuming only mount-site choices."""
        mount_choices = copy.copy(mount_choices)
        base = mjcf.from_path(components[root_component_name])
        unresolved_component_mounts: list[mjcf.RootElement] = [base]
        while len(unresolved_component_mounts) > 0:
            comp = unresolved_component_mounts.pop(0)
            for mount_site in comp.find_all("site"):
                mount_site_name = getattr(mount_site, "name", None)
                if not isinstance(mount_site_name, str) or not mount_site_name.endswith(
                    "_mount"
                ):
                    continue
                component_options = mount_site_name.split(":")[-1]
                component_options = component_options.split("_mount")[0].split("|")
                if len(component_options) > 1:
                    # Mirror grow_components_with_params: consume a choice only
                    # when there are multiple options.
                    component_option_idx = mount_choices.pop(0) % len(component_options)
                else:
                    component_option_idx = 0
                child_name = component_options[component_option_idx]
                child = mjcf.from_path(components[child_name])
                child = resolve_actuators(child)
                mount_site.attach(child)
                unresolved_component_mounts.append(child)
        if len(mount_choices) != 0:
            raise ValueError("parameters overspecified")
        return base

    def compute_independent_choice_counts(
        root: mjcf.RootElement,
    ) -> list[int]:
        """Return per-consumed-choice option counts, respecting constraints."""
        variation_constraints: dict[str, set[str]] = {}
        for variation_constraint in root.find_all("text"):
            variation_constraints[variation_constraint.name] = set(
                variation_constraint.data.split(";")
            )

        resolved_variations: set[str] = set()
        counts: list[int] = []

        for v in root.find_all("numeric"):
            field_name = v.name
            prefix = "/".join(v.full_identifier.split("/")[:-1])
            if field_name in resolved_variations:
                continue

            try:
                (
                    identifier,
                    namespace,
                    attr_name,
                    idx,
                    randomization_type,
                    _value_resolver_lambda,
                ) = field_name.split("\\")
            except ValueError:
                # Unexpected format; skip rather than risking overspecifying.
                continue

            # Mirror resolve_variations_with_params element lookup behavior.
            element = root.find(
                namespace=namespace,
                identifier=os.path.join(prefix, identifier).replace(
                    "@", PREFIX_SEPARATOR
                ),
            )
            if element is None:
                if STRICT_RESOLUTION:
                    raise ValueError(f"Element not found: {namespace} {identifier}")
                continue

            attr = getattr(element, attr_name)
            attr_is_array = isinstance(attr, np.ndarray)
            if attr_is_array:
                if idx != ":":
                    original_value = attr[int(idx)]
                else:
                    original_value = attr[:]
            else:
                original_value = attr

            randomization_data = v.data
            if attr_is_array:
                randomization_data = randomization_data.reshape(
                    -1, *np.asarray(original_value).shape
                )

            if randomization_type == "choice":
                num_choices = int(len(randomization_data))
                if num_choices > 1:
                    counts.append(num_choices)

            # Mark this field resolved, and if it belongs to a constraint group,
            # mark the rest of the group resolved too (no additional params).
            resolved_variations.add(field_name)
            for constraint_group in variation_constraints.values():
                if field_name in constraint_group:
                    for other_field_name in constraint_group:
                        resolved_variations.add(other_field_name)

        return counts

    # Base case: no mount sites left -> enumerate variation choices on full model.
    if len(unresolved_mount_sites) == 0:
        root = build_fully_mounted_model(
            root_component_name=_root_component_name,
            mount_choices=current_choices,
        )

        # For uniforms we return an upper bound (extra uniforms are OK).
        uniform_names = [
            numeric.name
            for numeric in root.find_all("numeric")
            if "\\choice\\" not in numeric.name
        ]

        # Enumerate the remaining (discrete) variation choices exactly.
        choice_counts = compute_independent_choice_counts(root)
        if len(choice_counts) == 0:
            return [(current_choices, uniform_names)]
        return recurse_choice(
            current_choices=copy.copy(current_choices),
            current_uniforms=uniform_names,
            unresolved_choices=choice_counts,
        )

    mount_site = unresolved_mount_sites.pop(0)
    mount_name = getattr(mount_site, "name", None)
    if not isinstance(mount_name, str):
        raise ValueError("Mount site missing string name")
    component_options = mount_name.split(":")[-1].split("_mount")[0].split("|")

    if len(component_options) > 1:
        for option_idx, option_name in enumerate(component_options):
            new_choices = current_choices + [option_idx]
            results.extend(
                recurse_component_resolve_after_mount(
                    option_name,
                    copy.copy(new_choices),
                    copy.copy(current_uniforms),
                    copy.copy(unresolved_mount_sites),
                    copy.copy(unresolved_choices),
                    components,
                    _root_component_name=_root_component_name,
                )
            )
    else:
        option_name = component_options[0]
        results.extend(
            recurse_component_resolve_after_mount(
                option_name,
                copy.copy(current_choices),
                copy.copy(current_uniforms),
                copy.copy(unresolved_mount_sites),
                copy.copy(unresolved_choices),
                components,
                _root_component_name=_root_component_name,
            )
        )

    if len(results) == 0:
        return [(current_choices, current_uniforms)]

    return results


def enumerate_choices(
    components: dict[str, str],
    start_component: str = "base",
    resolve_before_mount: bool = False,
) -> list[tuple[list[int], int]]:
    # returns a list of all possible choices (list of ints) along with how many continous variations
    # there are for that sequence of choices
    if resolve_before_mount:
        choices = recurse_component_resolve_before_mount(
            start_component, [], [], [], [], components
        )
    else:
        choices = recurse_component_resolve_after_mount(
            start_component, [], [], [], [], components
        )
    return [(c[0], len(c[1])) for c in choices]


def _grow_components_from_choices_and_seed(
    seed: int,
    choices: list[int],
    num_uniforms: int,
    *,
    components: dict[str, str],
    start_component: str,
    gravity_compensation: bool,
    resolve_before_mount: bool,
) -> mjcf.RootElement:
    rs = np.random.RandomState(seed)
    return grow_components_with_params(
        components=components,
        choices=choices,
        uniforms=rs.uniform(0, 1, num_uniforms).tolist(),
        start_component=start_component,
        gravity_compensation=gravity_compensation,
        resolve_before_mount=resolve_before_mount,
    )


def make_robot(
    components: dict[str, str],
    *,
    gravity_compensation: bool = False,
    resolve_before_mount: bool = False,
    start_component: str = "base",
):
    """Bind a components dict to (seed_fn, from_params_fn, from_choices_fn).

    Returned callables are `functools.partial` over module-level helpers so
    they pickle with stdlib `pickle` (required by `multiprocessing.Pool`,
    which `cma`'s `n_jobs>0` path uses).
    """
    grow_kwargs = dict(
        gravity_compensation=gravity_compensation,
        resolve_before_mount=resolve_before_mount,
        start_component=start_component,
    )

    seed_fn = functools.partial(
        grow_components,
        components=components,
        **grow_kwargs,
    )
    from_params_fn = functools.partial(
        grow_components_with_params,
        components=components,
        **grow_kwargs,
    )
    from_choices_fn = functools.partial(
        _grow_components_from_choices_and_seed,
        components=components,
        **grow_kwargs,
    )
    return seed_fn, from_params_fn, from_choices_fn
