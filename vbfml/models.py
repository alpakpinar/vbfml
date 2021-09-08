from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def sequential_dense_model(
    n_features: int, n_layers: int, n_nodes: "list[int]", n_classes: int
) -> Sequential:
    model = Sequential()

    assert (
        len(n_nodes) == n_layers
    ), "Inconsistent number of layers and node specification!"

    model.add(
        Dense(
            n_nodes[0],
            input_dim=n_features,
            activation="relu",
        )
    )
    for ilayer in range(1, n_layers):
        model.add(
            Dense(
                n_nodes[ilayer],
                activation="relu",
            )
        )
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
