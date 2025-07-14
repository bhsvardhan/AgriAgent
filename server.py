# fl_server.py

import flwr as fl

if __name__ == "__main__":
    # Start the FL server with FedAvg strategy
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg()
    )
