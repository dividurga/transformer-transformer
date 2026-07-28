import numpy as np
import pytest

from t2.robogen.components import enumerate_choices
from t2.robogen.umi_on_legs_plus_plus import (
    COMPONENTS,
    umi_on_legs_plus_plus_from_params,
)


def test_robogen_enumeration():
    choices = enumerate_choices(COMPONENTS)
    for choice, num_uniforms in choices:
        # this is an upper bound,on number of random uniform variables
        # since sometimes uniform variables can't be resolved
        # when STRICT_RESOLUTION is false
        rs = np.random.RandomState(0)
        umi_on_legs_plus_plus_from_params(
            choices=list(choice),
            uniforms=rs.uniform(0, 1, num_uniforms).tolist(),
        )

        with pytest.raises(IndexError):
            # too few choices should raise an error
            umi_on_legs_plus_plus_from_params(
                choices=list(choice)[:-1],
                uniforms=rs.uniform(0, 1, num_uniforms).tolist(),
            )
        with pytest.raises(ValueError):
            # too many choices should raise an error
            umi_on_legs_plus_plus_from_params(
                choices=list(choice) + [0],
                uniforms=rs.uniform(0, 1, num_uniforms).tolist(),
            )
