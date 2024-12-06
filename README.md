# TNO PET Lab - Federated Learning (FL) - Protocols - Logistic Regression

Implementation of a Federated Learning scheme for Logistic Regression. This
library was designed to facilitate both developers that are new to cryptography
and developers that are more familiar with cryptography.

Supports:

- Any number of clients and one server.
- Horizontal fragmentation
- Binary classification (multi-class not yet supported)
- Both fixed learning rate or second-order methods (Hessian)

### PET Lab

The TNO PET Lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of PET solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed PET functionalities to boost the development of new protocols and solutions.

The package `tno.fl.protocols.logistic_regression` is part of the [TNO Python Toolbox](https://github.com/TNO-PET).

_Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws._  
_This implementation of cryptographic software has not been audited. Use at your own risk._

## Documentation

Documentation of the `tno.fl.protocols.logistic_regression` package can be found
[here](https://docs.pet.tno.nl/fl/protocols/logistic_regression/1.1.0).

## Install

Easily install the `tno.fl.protocols.logistic_regression` package using `pip`:

```console
$ python -m pip install tno.fl.protocols.logistic_regression
```

_Note:_ If you are cloning the repository and wish to edit the source code, be
sure to install the package in editable mode:

```console
$ python -m pip install -e 'tno.fl.protocols.logistic_regression'
```

If you wish to run the tests you can use:

```console
$ python -m pip install 'tno.fl.protocols.logistic_regression[tests]'
```

## Usage

This package uses federated learning for training a logistic regression model on
datasets that are distributed amongst several clients. Below is first a short
overview of federated learning in general and how this has been implemented in
this package. In the next section, a minimal working example is provided. This
code is also available in the repository in the `examples` folder.

### Federated Learning

In Federated Learning, several clients, each with their own data, wish to fit a
model on their combined data. Each client computes a local update on their model
and sends this update to a central server. This server combines these updates,
updates the global model from this aggregated update and sends this new model
back to the clients. Then the process repeats: the clients compute the local
updates on this new model, send this to the server, which combines it and so on.
This is done until the server notices that the model has converged.

This package implements binary logistic regression. So each client has a data
set, that contains data and for each row a binary indicator. The goal is to
predict the binary indicator for new data. For example, the data could images of
cats and dogs and the binary indicator indicates whether it is a cat or a dog.
The goal of the logistic regression model, is to predict for new images whether
it contains a cat or a dog. More information on logistic regression is widely
available.

In the case of logistic regression, the updates the client compute consist of a
gradient. This model also implements a second-order derivative (Newton's
method).

### Implementation

The implementation of federated logistic regression consist of two classes with
the suggestive names `Client` and `Server`. Each client is an instance of
`Client` and the server is an instance of the `Server` class. These classes are
passed the required parameters and a communication pool.
Calling the `.run` method with the data will perform the federated learning
and returns the resulting logistic regression model (as a numpy array).

#### Communication

The client and the servers must be given a communication pool during initialization.
This is a `Pool` object from the `tno.mpc.communication` package, which is also part
of the PET lab. It is used for the communication amongst the server and the
clients. We refer to this package for more information about this.
The example file also gives an example of how to set up a simple communication pool.

Since the communication package uses `asyncio` for asynchronous handling, this
federated learning package depends on it as well. For more information about
this, we refer to the
[tno.mpc.communication documentation](https://docs.pet.tno.nl/mpc/communication/)

#### Passing the data

Once the client and the server have been properly initialized,
the federated learning can be performed using the `.run()` function.
This function has two arguments.
The first is a numpy array containing the covariates / training data.
The second is another numpy array of booleans containing the target data.
So the first one contains the sample data and the second contains the category the sample belongs to.
Currently, only binary classification is supported.

#### Other customization

All settings are passed as parameters to the client and the server.
This includes:

- **fit_intercept:** Should an intercept column be added to the data as first column. Default: False
- **max_iter:** The maximum number of iterations in the learning process. Default: 25.
- **server_name:** The name of the server handler in the pool object. Default: 'server'.

In addition, there are many possibilities for overriding client/server functions,
such as a preprocessing function, computing the client weights, or the initial model.

### Example code

Below is a very minimal example of how to use the library. It consists of two
clients, Alice and Bob, who want to fit a model for recognizing the setosa iris
flower. Below is an excerpt from their data sets:

`data_alice.csv`

```csv
sepal_length,sepal_width,petal_length,petal_width,is_setosa
5.8,2.7,5.1,1.9,0
6.9,3.1,5.4,2.1,0
5,3.4,1.5,0.2,1
5.2,4.1,1.5,0.1,1
6.7,3.1,5.6,2.4,0
6.3,2.9,5.6,1.8,0
5.6,2.5,3.9,1.1,0
5.7,3.8,1.7,0.3,1
5.8,2.6,4,1.2,0
```

`data_bob.csv`

```csv
sepal_length,sepal_width,petal_length,petal_width,is_setosa
7.2,3,5.8,1.6,0
6.7,2.5,5.8,1.8,0
6,3.4,4.5,1.6,0
4.8,3.4,1.6,0.2,1
7.7,3.8,6.7,2.2,0
5.4,3.9,1.3,0.4,1
7.7,3,6.1,2.3,0
7.1,3,5.9,2.1,0
6.1,2.9,4.7,1.4,0
```

We create the following code to run the federated learning algorithm:

`main.py`

```python
"""
This module runs the logistic regression protocol on an example data set.
By running the script three times with command line argument 'server', 'alice'
and 'bob' respectively, you can get a demonstration of how it works.
"""

import asyncio
import sys

import pandas as pd

from tno.mpc.communication import Pool

from tno.fl.protocols.logistic_regression.client import Client
from tno.fl.protocols.logistic_regression.server import Server


async def run_client(name: str, port: int) -> None:
    # Create Pool
    pool = Pool()
    pool.add_http_server(addr="localhost", port=port)
    pool.add_http_client(name="server", addr="localhost", port=8080)
    # Get Data
    csv_data = pd.read_csv("data_" + name + ".csv")
    data = csv_data[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].to_numpy()
    target = csv_data["is_setosa"].to_numpy()
    # Create Client
    client = Client(pool, fit_intercept=True, max_iter=10)
    print(await client.run(data, target))


async def run_server() -> None:
    # Create Pool
    pool = Pool()
    pool.add_http_server(addr="localhost", port=8080)
    pool.add_http_client(name="alice", addr="localhost", port=8081)
    pool.add_http_client(name="bob", addr="localhost", port=8082)
    # Create Client
    server = Server(pool, max_iter=10)
    await server.run()


async def async_main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Player name must be provided.")
    if sys.argv[1].lower() == "server":
        await run_server()
    elif sys.argv[1].lower() == "alice":
        await run_client("alice", 8081)
    elif sys.argv[1].lower() == "bob":
        await run_client("bob", 8082)
    else:
        raise ValueError(
            "This player has not been implemented. Possible values are: server, alice, bob"
        )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_main())
```

To run this script, call `main.py` from the folder where the data files and the
config file are located. As command line argument, pass it the name of the party
running the app: 'Alice', 'Bob', or 'Server'. To run in on a single computer,
run the following three command, each in a different terminal: Note that if a
client is started prior to the server, it will throw a ClientConnectorError.
Namely, the client tries to send a message to port the server, which has not
been opened yet. After starting the server, the error disappears.

```commandline
python main.py alice
python main.py bob
python main.py server
```

The output for the clients will be something similar to:

```commandline
>>> python main.py alice
2024-01-18 16:01:56,735 - tno.mpc.communication.httphandlers - INFO - Serving on localhost:8081
2024-01-18 16:01:58,655 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,655 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,671 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,693 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,709 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,709 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,724 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,740 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,756 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,771 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
2024-01-18 16:01:58,793 - tno.mpc.communication.httphandlers - INFO - Received message from 127.0.0.1:8080
[[-7.63901840925708], [2.985418690990691], [4.688929649931743], [-6.397069834606601], [-6.008454039386442]]
```

We first see the client setting up the connection with the server. Then we have
ten rounds of training, as indicated in the configuration file. Finally, we
print the resulting model. We obtain the following coefficients for classifying
setosa irises:

| Parameter    | Coefficient        |
| ------------ | ------------------ |
| intercept    | -7.63901840925708  |
| sepal_length | 2.985418690990691  |
| sepal_width  | 4.688929649931743  |
| petal_length | -6.397069834606601 |
| petal_width  | -6.008454039386442 |
