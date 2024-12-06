"""
Server module for logistic regression
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from tno.mpc.communication import Pool

from tno.fl.protocols.logistic_regression.msg_ids import (
    COEFS,
    INIT_MODEL,
    LOCAL_HESSIAN,
    LOCAL_UPDATE,
    STANDARD_ERROR,
    WEIGHT,
)

ModelType = npt.NDArray[np.float64]
GradientType = npt.NDArray[np.float64]
HessianType = npt.NDArray[np.float64]
UpdateType = tuple[GradientType, HessianType]


class Server:
    """
    The Server class. Responsible for aggregating results of the clients.
    """

    def __init__(self, pool: Pool, max_iter: int = 25) -> None:
        """
        Initializes the server.

        :param pool: The communication pool.
        :param max_iter: The max number of epochs
        """
        # Set the communication pool
        self.pool = pool
        # Set the n_epochs
        self.max_iter = max_iter

    async def _get_weights_from_clients(self) -> dict[str, float]:
        """
        Receive the weights from all clients.

        :return: A dictionary containing the weights. Each key is a client id and its value is the
            corresponding weight.
        """
        return {
            client: int(n_rows)
            for client, n_rows in await self.pool.recv_all(msg_id=WEIGHT)
        }

    async def _receive_updates(self) -> dict[str, UpdateType]:
        """
        Receive updates from the clients and put them in a dictionary.

        :return: A dictionary containing the updates. Each key is a client id
            and its value is the corresponding update (gradient & hessian) from that client.
        """
        return dict(await self.pool.recv_all(msg_id=LOCAL_UPDATE))

    @staticmethod
    def _weighted_average(
        values: dict[str, npt.NDArray[np.float64]],
        weights: dict[str, float],
    ) -> npt.NDArray[np.float64]:
        """
        Take the weighted averages of values. Matches the values and weights by the keys in the dicts.

        :param values: Dictionary containing clients as keys and their values.
        :param weights: Dictionary containing clients as keys and their weights as values.
        :return: A weighted average of the gradients.
        """
        # Match the weights and values by keys
        value_weights = [(values[client], weights[client]) for client in values.keys()]
        values_sorted, weights_sorted = list(zip(*value_weights))

        # Take weighted average of gradients
        average: npt.NDArray[np.float64] = np.average(
            np.stack(values_sorted), axis=0, weights=weights_sorted
        )
        return average

    async def _aggregate(
        self,
        update_per_client: dict[str, UpdateType],
        weights_per_client: dict[str, float],
    ) -> UpdateType:
        """
        Aggregate gradients by taking weighted average. Works for gradients of any order.
        First sorts the gradients and weights in order, then averages the gradients based on their
        weights.

        :param update_per_client: Dictionary containing clients as keys and their gradient as
            values.
        :param weights_per_client: Dictionary containing clients as keys and their weights as
            values.
        :return: A weighted average of the gradients.
        """
        # Sort the gradients and weights in order
        gradient_per_client = {k: v[0] for k, v in update_per_client.items()}
        gradient = self._weighted_average(gradient_per_client, weights_per_client)
        hessian_per_client = {k: v[1] for k, v in update_per_client.items()}
        hessian = self._weighted_average(hessian_per_client, weights_per_client)
        return gradient, hessian

    @staticmethod
    def _update_model(model: ModelType, update: UpdateType) -> ModelType:
        """ "
        Update model according to gradient and hessian.

        :param model: The initial model
        :param update: The update
        :return: The model updated according to the gradient and the hessian
        """
        gradient, hessian = update
        return model - cast(
            npt.NDArray[np.float64], np.dot(np.linalg.inv(hessian), gradient).flatten()
        )

    async def _run_epoch(
        self, weights_per_client: dict[str, float], model: ModelType
    ) -> ModelType:
        """
        Perform one epoch.

        :param weights_per_client: Dictionary containing clients as keys
            and their weights as values.
        :param model: The model at the start of the epoch.
        :return: The updated model after the epoch.
        """
        # Receive and aggregate updates
        updates = await self._receive_updates()
        update = await self._aggregate(updates, weights_per_client)

        # Update model according to gradients and hessians
        updated_model = self._update_model(model, update)

        # Distribute updated model
        await self.pool.broadcast(updated_model, COEFS)

        return updated_model

    async def run(self) -> None:
        """
        Runs the entire learning process.
        """
        # Get the weights per client
        weights_per_client = await self._get_weights_from_clients()

        # Get initial model
        init_models = dict(await self.pool.recv_all(msg_id=INIT_MODEL))
        model = self._weighted_average(init_models, weights_per_client)
        await self.pool.broadcast(model, msg_id=INIT_MODEL)

        for _epoch in range(self.max_iter):
            model = await self._run_epoch(weights_per_client, model)

    async def compute_standard_error(self) -> None:
        """
        Compute the standard error for the clients.
        """
        # Receive all the weights and hessians from clients
        messages = dict(await self.pool.recv_all(msg_id=LOCAL_HESSIAN))
        weights = {key: message[0] for key, message in messages.items()}
        hessians = {key: message[1] for key, message in messages.items()}
        # Compute aggregated gradient and standard error
        hessian = self._weighted_average(hessians, weights)
        standard_errors = np.sqrt(np.linalg.inv(hessian).diagonal())
        # Return standard error
        await self.pool.broadcast(standard_errors, msg_id=STANDARD_ERROR)

    async def compute_statistics(self) -> None:
        """
        Help in computing the statistics for the clients.
        At this moment, the server is only needed in calculating the standard error.
        This function is added to the same function can be called on both client and server.
        """
        await self.compute_standard_error()
