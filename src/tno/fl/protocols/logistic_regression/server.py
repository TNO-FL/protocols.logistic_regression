"""
Server module for logistic regression
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tno.mpc.communication import Pool

from . import msg_ids
from .config import Config


class Server:
    """
    The Server class. Responsible for aggregating results of the clients.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the server by setting the config and initializing the communication pool.

        :param config: The configuration for the experiment
        """
        self.config = config
        self.pool = self.create_communication_pool()

    def create_communication_pool(self) -> Pool:
        """
        Create a communication pool with all the clients.

        :return: The communication pool
        """
        pool = Pool()
        pool.add_http_server(
            addr=self.config.server.address, port=self.config.server.port
        )
        for client in self.config.clients:
            pool.add_http_client(client.name, addr=client.address, port=client.port)
        return pool

    async def preprocessing(self) -> None:
        """
        Server's role in data preprocessing. No action required in default implementation.
        """

    async def get_weights_from_clients(self) -> dict[str, int]:
        """
        Receive the weights from all clients.

        :return: A dictionary containing the weights. Each key is a client id and its value is the
            corresponding weight.
        """
        return {
            client: int(n_rows)
            for client, n_rows in await self.pool.recv_all(msg_id=msg_ids.WEIGHT)
        }

    def initial_model(self) -> npt.NDArray[np.float64]:
        """
        Returns an initialized model.

        :return: An initialized model (all zeros).
        """
        n_coefs = len(self.config.data_columns) + int(self.config.intercept)
        return np.zeros((n_coefs, 1))

    async def get_gradients_from_clients(
        self, msg_id: str
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Receive gradients from the clients and put them in a dictionary.

        :param msg_id: The message id for receiving the gradients.
        :return: A dictionary containing the gradients. Each key is a client id
            and its value is the corresponding gradient from that client.
        """
        return dict(await self.pool.recv_all(msg_id=msg_id))

    @staticmethod
    async def aggregate(
        gradient_per_client: dict[str, npt.NDArray[np.float64]],
        weights_per_client: dict[str, int],
    ) -> npt.NDArray[np.float64]:
        """
        Aggregate gradients by taking weighted average. Works for gradients of any order.
        First sorts the gradients and weights in order, then averages the gradients based on their
        weights.

        :param gradient_per_client: Dictionary containing clients as keys and their gradient as
            values.
        :param weights_per_client: Dictionary containing clients as keys and their weights as
            values.
        :return: A weighted average of the gradients.
        """
        # Sort the gradients and weights in order
        weighted_gradients = [
            (gradient_per_client[client], weights_per_client[client])
            for client in gradient_per_client.keys()
        ]
        gradients, weights = list(zip(*weighted_gradients))
        # Take weighted average of gradients
        average: npt.NDArray[np.float64] = np.average(
            np.stack(gradients), axis=0, weights=weights
        )
        return average

    async def run(self) -> npt.NDArray[np.float64]:
        """
        Runs the entire learning process.

        :return: The outcome model.
        :raises ValueError: if the learning rate is not set.
        """
        # Preprocess the data
        await self.preprocessing()

        # Get the weights per client
        weights_per_client = await self.get_weights_from_clients()

        model = self.initial_model()
        for _epoch in range(self.config.n_epochs):
            # Take weighted average of gradients and hessians
            gradients = await self.get_gradients_from_clients(
                msg_id=msg_ids.LOCAL_GRADIENT
            )
            gradient = await self.aggregate(gradients, weights_per_client)

            # Update based on learning rate or hessian depending on config
            if self.config.hessian:
                hessians = await self.get_gradients_from_clients(
                    msg_id=msg_ids.LOCAL_HESSIAN
                )
                hessian = await self.aggregate(hessians, weights_per_client)
                model -= np.dot(np.linalg.inv(hessian), gradient)
            elif self.config.learning_rate is not None:
                model -= gradient * self.config.learning_rate
            else:
                raise ValueError("Learning rate is not set.")

            # Distribute updated coefficients
            await self.pool.broadcast(model, msg_ids.COEFS)

        return model
