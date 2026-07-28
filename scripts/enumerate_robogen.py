import subprocess

from t2.robogen.components import enumerate_choices
from t2.robogen.umi_on_legs_plus_plus import COMPONENTS

if __name__ == "__main__":
    output_path = "umi_on_legs_plus_plus_robotokens/"
    choices = enumerate_choices(COMPONENTS)
    for choice, num_uniforms in sorted(choices, key=lambda x: str(x[0])):
        choice_str = "[" + ",".join(map(str, choice)) + "]"
        output_name = "".join(map(str, choice)) + ".zarr"
        cmd = [
            "python",
            "scripts/enumerate_robogen_robotokens.py",
            "num_hardware=100",
            "num_processes=32",
            f"output_path={output_path}/{output_name}",
            f"choices={choice_str}",
            f"num_uniforms={num_uniforms}",
        ]
        subprocess.run(cmd, check=True)
