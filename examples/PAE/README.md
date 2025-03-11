# Probabilistic AutoEncoder

The core of `SuPAErnova` is a probabilistic autoencoder trained to encode a SN Ia spectrum into a set of latent parameters, and decode a set of latent parameters into a SN Ia spectrum.

## Encoder Design

### Dense Encoder

```mermaid
graph LR
    IN1[["
        Input
        Spectra Fluxes
        [max_n_spec, n_wave]
        Float32
    "]]
    IN2[["
        Input
        Specta Phases
        [max_n_spec, 1]
        Float32
    "]]
    IN3[["
        Input
        Spectra Masks
        [max_n_spec, n_wave]
        Int32
    "]]

    %% Concatenate inputs
    X1("
        Concatenate
        [max_n_spec, n_wave + 1]
        Float32
    ")

    IN1-->X1
    IN2-->X1

    DENSEI["
        Dense
        [max_n_spec, n]
        Float32
        activation = activation
        kernel_regularizer = kernel_regulariser
    "]

    X1-->|for n in encode_dims|DENSEI

    DROPOUT("
        Dropout
        [max_n_spec, n]
        Float32
        rate = dropout
        noise_shape = [None, 1, None]
    ")

    BATCHNORM("
        BatchNormalization
        [max_n_spec, n]
        Float32
    ")

    DENSEI-->|if dropout > 0|DROPOUT
    DENSEI-->|if batchnorm|BATCHNORM

    DENSE_SPEC["
        Dense
        [max_n_spec, max_n_spec]
        Float32
        activation = activation
        kernel_regularizer = kernel_regulariser
    "]

    DROPOUT-->|endfor|DENSE_SPEC
    BATCHNORM-->|endfor|DENSE_SPEC
    DENSEI-->|endfor|DENSE_SPEC
```
