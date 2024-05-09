from activation import gelu, relu, silu, swiglu
from embedding import absolute_sinusoidal as sinusoidal
from normalization import deepnorm, layernorm, rmsnorm, softmax

if __name__ == "__main__":

    gelu.benchmark.run(print_data=True)
    relu.benchmark.run(print_data=True)
    silu.benchmark.run(print_data=True)
    swiglu.benchmark.run(print_data=True)

    sinusoidal.benchmark.run(print_data=True)

    deepnorm.benchmark.run(print_data=True)
    layernorm.benchmark.run(print_data=True)
    rmsnorm.benchmark.run(print_data=True)
    softmax.benchmark.run(print_data=True)
