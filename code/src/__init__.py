import tomli

from . import model


def run_scenario(scenario):
    """
    Run a single model scenario and return the results.

    :param scenario: A dictionary of scenario settings, including parameters.
    :type scenario: Dict[str, Any]
    """
    parameters = scenario["parameters"]
    x0 = parameters["x0"]
    n = parameters["num_samples"]
    seed = parameters["rng_seed"]
    return model.run_model(x0, n, seed)


def load_scenarios(toml_file):
    """
    Load model scenarios from a TOML file.

    :param toml_file: The filename from which to read the scenarios.
    :returns: A dictionary that maps scenario names to model parameters.
    """
    with open(toml_file, "rb") as f:
        data = tomli.load(f)

    return data["scenarios"]
