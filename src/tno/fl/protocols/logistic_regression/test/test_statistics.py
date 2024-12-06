"""
Test for the statistical functions
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from tno.mpc.communication import Pool

from tno.fl.protocols.logistic_regression.client import Client
from tno.fl.protocols.logistic_regression.server import Server


@pytest.mark.asyncio
async def test_statistics(http_pool_trio: tuple[Pool, Pool, Pool]) -> None:
    """
    Run the Cox regression with the Rotterdam example

    :param http_pool_trio: communication pool fixture for 3 parties
    """
    # Load the data
    csv_data_alice = pd.read_csv("examples/iris/data_alice.csv")
    data_alice = csv_data_alice[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].to_numpy()
    target_alice = csv_data_alice["is_setosa"].to_numpy()

    csv_data_bob = pd.read_csv("examples/iris/data_bob.csv")
    data_bob = csv_data_bob[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].to_numpy()
    target_bob = csv_data_bob["is_setosa"].to_numpy()

    # Set a model
    model = np.array([0.90271, 4.91107, -5.17815, -6.97310])

    # Run the protocol
    server = Server(http_pool_trio[0], max_iter=10)
    client1 = Client(http_pool_trio[1], max_iter=10, server_name="local0")
    client2 = Client(http_pool_trio[2], max_iter=10, server_name="local0")

    statistics: list[list[dict[str, float]] | None] = await asyncio.gather(
        *[
            server.compute_statistics(),
            client1.compute_statistics(data_alice, target_alice, model),
            client2.compute_statistics(data_bob, target_bob, model),
        ]
    )

    # Parse results
    assert statistics[1] is not None
    assert statistics[2] is not None
    standard_errors = [stat["se"] for stat in statistics[1]]
    z_values = [stat["z"] for stat in statistics[1]]
    p_values = [stat["p"] for stat in statistics[1]]

    standard_errors2 = [stat["se"] for stat in statistics[2]]
    z_values2 = [stat["z"] for stat in statistics[2]]
    p_values2 = [stat["p"] for stat in statistics[2]]

    # Tests
    assert np.array_equal(standard_errors, standard_errors2)
    assert np.array_equal(z_values, z_values2)
    assert np.array_equal(p_values, p_values2)
    assert np.isclose(
        standard_errors,
        [46.144631, 67.512293, 62.497795, 139.044876],
        rtol=1e-05,
        atol=1e-08,
    ).all()
    assert np.isclose(
        z_values, [0.019562622, 0.072743, -0.082853, -0.05015], rtol=1e-05, atol=1e-08
    ).all()
    assert np.isclose(
        p_values,
        [0.984392, 0.942010, 0.933968, 0.960002],
        rtol=1e-05,
        atol=1e-08,
    ).all()
