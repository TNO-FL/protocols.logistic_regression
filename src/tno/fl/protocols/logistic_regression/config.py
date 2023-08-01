"""Configuration parsing and storage for logistic regression"""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Party:
    """
    Class representing a party. Contains the data required by the communication pool.

    :param name: The name of the party
    :param address: The network address of the party
    :param port: The network port of the party
    :param train_data_path: The path to the training data
    """

    name: str = field(compare=False)
    address: str
    port: int
    train_data_path: Path = field(compare=False)

    def __str__(self) -> str:
        return f"{self.address}:{self.port}"

    def __hash__(self) -> int:
        return hash(str(self))


class ConfigParser:  # pylint: disable=too-many-instance-attributes
    """
    Parser for config file.

    :param LIST_SEPARATOR: the separator used for lists in the config file.
    :param EXPERIMENT_SECTION: The section name for experiment config.
    :param EXPERIMENT_DATA_COLUMNS_KEY: Key for which columns of data file should be used in
        training data.
    :param EXPERIMENT_TARGET_COLUMN_KEY: Key for which column contains target values.
    :param EXPERIMENT_INTERCEPT_KEY: Key for value whether intercept is used.
    :param EXPERIMENT_EPOCHS_KEY: Key for number of epochs
    :param EXPERIMENT_LEARNING_RATE_KEY: Key for learning rate (number or hessian)
    :param PARTY_SECTION: The section name for the parties section
    :param SERVER_NAME_KEY: Key for name of the server
    :param CLIENT_NAMES_KEY: Key for the list of client names
    :param PARTY_ADDRESS_KEY: Key for the network address of party
    :param PARTY_PORT_KEY: Key for network party of party
    :param PARTY_TRAIN_DATA_KEY: Key for training data location for party
    """

    LIST_SEPARATOR = ","

    EXPERIMENT_SECTION = "Experiment"
    EXPERIMENT_DATA_COLUMNS_KEY = "data_columns"
    EXPERIMENT_TARGET_COLUMN_KEY = "target_column"
    EXPERIMENT_INTERCEPT_KEY = "intercept"
    EXPERIMENT_EPOCHS_KEY = "n_epochs"
    EXPERIMENT_LEARNING_RATE_KEY = "learning_rate"

    PARTY_SECTION = "Parties"
    SERVER_NAME_KEY = "server"
    CLIENT_NAMES_KEY = "clients"
    PARTY_ADDRESS_KEY = "address"
    PARTY_PORT_KEY = "port"
    PARTY_TRAIN_DATA_KEY = "train_data"

    def __init__(self, config_file_path: Path) -> None:
        """
        Load the config file and parse the objects variables from the file.

        :param config_file_path: The path to the config file
        """
        self.config = configparser.ConfigParser()
        self.load_config(config_file_path)

        self.server = self.parse_server()
        self.clients = self.parse_clients()
        self.data_columns = self.parse_data_columns()
        self.target_column = self.parse_target_column()
        self.intercept = self.parse_intercept()
        self.n_epochs = self.parse_n_epochs()
        self.learning_rate, self.hessian = self.parse_learning_rate()

    def load_config(self, config_file_path: Path) -> None:
        """
        Load the config file.

        :param config_file_path: The path to the config file
        """
        self.config.read(config_file_path)

    def parse_server(self) -> Party:
        """
        Parse the server from the config.

        :return: the server from the config file
        """
        server_name = self.config.get(self.PARTY_SECTION, self.SERVER_NAME_KEY)
        return self.parse_party(server_name)

    def parse_clients(self) -> list[Party]:
        """
        Parse the list of clients from the config.

        :return: the list of clients from the config file
        """
        party_names = self.parse_list(self.PARTY_SECTION, self.CLIENT_NAMES_KEY)
        return [self.parse_party(name) for name in party_names]

    def parse_party(self, name: str) -> Party:
        """
        Parse party with given name from config file.

        :param name: the name of the party to parse
        :return: the list of clients from the config file
        """
        address = self.config.get(name, self.PARTY_ADDRESS_KEY)
        port = self.config.getint(name, self.PARTY_PORT_KEY)
        train_data = Path(self.config.get(name, self.PARTY_TRAIN_DATA_KEY, fallback=""))
        return Party(name=name, address=address, port=port, train_data_path=train_data)

    def parse_data_columns(self) -> list[str]:
        """
        Parse the data columns from config file.

        :return: The list of data columns names
        """
        return self.parse_list(
            self.EXPERIMENT_SECTION, self.EXPERIMENT_DATA_COLUMNS_KEY
        )

    def parse_target_column(self) -> str:
        """
        Parse the target column from config file.

        :return: The target column name
        """
        return self.config.get(
            self.EXPERIMENT_SECTION, self.EXPERIMENT_TARGET_COLUMN_KEY
        )

    def parse_intercept(self) -> bool:
        """
        Parse whether intercept column should be added according to config.

        :return: Whether intercept column should be added
        """
        return self.config.getboolean(
            self.EXPERIMENT_SECTION, self.EXPERIMENT_INTERCEPT_KEY
        )

    def parse_n_epochs(self) -> int:
        """
        Parse number of epochs from config.

        :return: number of epochs
        """
        return self.config.getint(self.EXPERIMENT_SECTION, self.EXPERIMENT_EPOCHS_KEY)

    def parse_learning_rate(self) -> tuple[float | None, bool]:
        """
        Parse learning rate from config.

        :return: First element of tuple is a boolean indicating whether a hessian must be used.
            Second element is the learning rate if applicable.
        :raises ValueError: if learning rate cannot be parsed.
        """
        learning_rate = self.config.get(
            self.EXPERIMENT_SECTION, self.EXPERIMENT_LEARNING_RATE_KEY
        )
        try:
            return float(learning_rate), False
        except ValueError as exc:
            if learning_rate == "hessian":
                return None, True
            raise ValueError("Cannot parse learning rate.") from exc

    def parse_list(self, section: str, key: str) -> list[str]:
        """
        Parse a config item that contains a list of values.

        :param section: Which [Section] in the config you want
        :param key: Which key you want within that section
        :return: List of values given for that key
        """

        return list(
            filter(None, self.config.get(section, key).split(self.LIST_SEPARATOR))
        )

    def to_config(self) -> Config:
        """
        Convert the parse to an actual config object.

        :return: the config object containing the data from the parsed config file
        """
        return Config(
            clients=self.clients,
            server=self.server,
            data_columns=self.data_columns,
            target_column=self.target_column,
            intercept=self.intercept,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            hessian=self.hessian,
        )


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """
    The configuration of the experiment.

    :param clients: List of clients
    :param server: Server party
    :param data_columns: Names of data columns
    :param target_column: Name of target column
    :param intercept: Whether to use intercept column
    :param n_epochs: Number of epochs
    :param learning_rate: Learning rate
    :param hessian: Whether to use second derivative
    """

    clients: list[Party]
    server: Party
    data_columns: list[str]
    target_column: str
    intercept: bool
    n_epochs: int
    learning_rate: float | None
    hessian: bool

    @staticmethod
    def from_file(config_path: Path) -> Config:
        """
        Create config from config file.

        :param config_path: The path to the file
        :return: The configuration according to config file
        """
        return ConfigParser(config_path).to_config()

    def get_client_by_name(self, client_name: str) -> Party:
        """
        Get a client by name.

        :param client_name: the name of the party which is requested
        :return: the client with name client_name
        :raises ValueError: if the client name is not in the list of clients
        """
        try:
            return next(client for client in self.clients if client.name == client_name)
        except StopIteration as exc:
            raise ValueError(f"Client {client_name} does not exist") from exc
