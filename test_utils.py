import utils


def test_hparams_combinations():
    # a test case to check all possible combinations of hyper parameters
    h_params_grid = {
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
        "C": [0.1, 1, 2, 5, 10],
    }
    h_param_combinations = utils.get_combinations_with_keys(h_params_grid)

    assert len(h_param_combinations) == len(h_params_grid["gamma"]) * len(
        h_params_grid["C"]
    )


def test_hparams_combinations_values():
    # a test case to check all possible combinations of hyper parameters values
    h_params_grid = {
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
        "C": [0.1, 1, 2, 5, 10],
    }
    h_param_combinations = utils.get_combinations_with_keys(h_params_grid)

    assert len(h_param_combinations) == len(h_params_grid["gamma"]) * len(
        h_params_grid["C"]
    )
    expected_parma_combo_1 = {"gamma": 0.001, "C": 1}
    expected_parma_combo_2 = {"gamma": 0.01, "C": 1}

    assert expected_parma_combo_1 in h_param_combinations
    assert expected_parma_combo_2 in h_param_combinations
