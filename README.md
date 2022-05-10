# Model architecture

`R(beta, pose)` function is designed as MLP. More specifically, we concatenated beta and pose as input then we obtain displacement using
4 fully connected layers with a dimensionality of 512 and ReLU activation function.

## Formulation
T(beta, pose) = T<sub>G</sub> + R(beta, pose)

M(beta, pose) = W(T(beta, pose), J(beta), W<sub>G</sub>)

- `T` - garment template

- `W` - skinning weights

- `R(beta, pose)` - displacement regressor

- `M(beta, pose)` - garment model

- `J(beta)` - joint locations

# Train
We used PBNS's loss functions without any changes for train the garment.