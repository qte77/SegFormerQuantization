"""
Following structure:
# scenarios
scenario_1_name:
  steps:
    step_1_name:
      from:
        <import_function>: <function>
        [import_N]: [function_N]
      docstring: "[docstring]"
      action:
        function: <function>
        args:
          [fun_arg_1]: Any
          [fun_arg_N]: Any
        returns:
          - [return_1]
          - [return_N]
      assert:
        [assert_return_1]: "Message_1"
        [assert_return_N]: "Message_N"
    step_N_name: ...
scenario_N_name:
  steps: ...
"""
from pathlib import Path
from typing import Dict
from yaml import safe_load

def parse_dsl(file_path: Path) -> Dict:
    """Parse and return DSL yaml."""
    with open(file_path, 'r') as file:
        config = safe_load(file)
    return config

def generate_test_file(
        scenario: Dict,
        test_file_path: Path
) -> None:
    """Generate a test file based on the scenario."""
    imports = "" # "import pytest"
    imports = "\n".join([
        f"from {k} import {v}"
        for step in scenario['steps']
        for k, v in scenario['steps'][step]['from'].items()
    ])
    
    for step in scenario['steps']:
        fun_used = scenario['steps'][step]['action']['function']
        fun_docstring = scenario['steps'][step]['action']['docstring']
        fun_args = scenario['steps'][step]['action']['args']
        fun_ret_ls = scenario['steps'][step]['action']['returns']
        asserts = scenario['steps'][step]['assert']

        if isinstance(fun_ret_ls, list):
            fun_returns = ", ".join(fun_ret_ls)
            fun_returns = f"{fun_returns} = "
        else:   
            fun_returns = ""

    with open(test_file_path, 'w') as test_file:
        for line in [
            f"{imports}\n",
            f"def test_{step.lower().replace(' ', '_')}():",
            f"\t\"\"\"{fun_docstring}\"\"\"",
            f"\tfun_args = {fun_args}",
            f"\t{fun_returns}{fun_used}(**fun_args)",
            *[f"\tassert {a}, \"{m}\"" for a,m in asserts.items()]
        ]:
            print(line)
            test_file.write(f"{line}\n")

def main():
    test_dir = Path('tests')
    dsl_dir = test_dir / 'DSL'
    dsl_format = '*.yaml'
    gen_test_path = test_dir / 'test_generated_from_DSL'
    gen_test_prefix = '_generated'

    if not dsl_dir.is_dir():
        raise ValueError(
            f"Expected directory does not exist: '{dsl_dir}'"
        )

    for dsl_file in dsl_dir.glob(dsl_format):
        scenarios = parse_dsl(dsl_file)
        if not isinstance(scenarios, Dict):
            raise ValueError(
                f"Expected type <dict|Dict> for scenarios, got {type(scenarios)}"
            )

    for scenario_name in scenarios:
        scenario = scenarios[scenario_name]
        if isinstance(scenario, Dict):
            test_case = scenario_name.lower().replace(' ', '_')
            test_file_name = f"test_{test_case}{gen_test_prefix}.py"
            test_file_path = gen_test_path / test_file_name
            print(f"\n:> generating {test_case} <:")
            generate_test_file(scenario, test_file_path)
        else:
            print(f"Expected type <dict|Dict> for scenario {scenario}, got {type(scenario)}")

if __name__ == "__main__":
    main()
