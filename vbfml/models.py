import numpy as np
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Flatten,
)

from tensorflow.keras.models import Sequential


def sequential_dense_model(
    n_features: int,
    n_layers: int,
    n_nodes: "list[int]",
    n_classes: int,
    dropout: float = 0,
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
        if dropout:
            model.add(Dropout(rate=dropout))
        model.add(
            Dense(
                n_nodes[ilayer],
                activation="relu",
            )
        )
    model.add(Dense(n_classes, activation="softmax"))
    return model


def sequential_convolutional_model(
    n_layers_for_conv: int,
    n_filters_for_conv: "list[int]",
    pool_size_for_conv: "list[int]",
    filter_size_for_conv: "list[int]",
    n_nodes_for_dense: "list[int]",
    n_layers_for_dense: int,
    n_classes: int,
    dropout: float = 0,
    image_shape: tuple = (40, 20, 1),
) -> Sequential:
    model = Sequential()

    assert (
        len(n_nodes_for_dense) == n_layers_for_dense
    ), "Inconsistent number of layers and node specification for dense network!"

    assert (len(n_filters_for_conv) == n_layers_for_conv) and (
        len(filter_size_for_conv) == n_layers_for_conv
    ), "Inconsistent number of layers and node specification for convolutional network!"

    # First, reshape the input image data
    num_pixels = np.prod(image_shape)
    model.add(Reshape(image_shape, input_shape=(num_pixels,)))

    for ilayer in range(n_layers_for_conv):
        model.add(
            Conv2D(
                n_filters_for_conv[ilayer],
                filter_size_for_conv[ilayer],
            )
        )
        model.add(MaxPooling2D(pool_size=pool_size_for_conv[ilayer]))

    model.add(Flatten())

    for ilayer in range(n_layers_for_dense):
        model.add(Dense(n_nodes_for_dense[ilayer], activation="relu"))
        if dropout:
            model.add(Dropout(rate=dropout))

    model.add(Dense(n_classes, activation="softmax"))
    return model
