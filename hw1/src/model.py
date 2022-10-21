from src.layer import Layer


class Model:

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
