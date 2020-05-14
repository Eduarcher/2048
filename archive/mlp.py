from sklearn.neural_network import MLPClassifier
import numpy as np


class MLPClassifierOverride(MLPClassifier):
    def __init__(self, expected_inputs, random_start_range=5, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=20000,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        self.expected_inputs = expected_inputs
        self.random_start_range = random_start_range

    def set_random_coefs(self):
        self.coefs_ = self.random_coefs()
        self.intercepts_ = self.random_intercepts()

    def set_coefs(self, coefs, intercepts):
        self.coefs_ = coefs
        self.intercepts_ = intercepts

    def random_coefs(self):
        layers = [self.expected_inputs]
        for layer in self.hidden_layer_sizes:
            layers.append(layer)
        layers.append(1)

        coefs = []
        for i in range(1, len(layers)):
            coefs.append((self.random_start_range * np.random.random_sample((layers[i-1], layers[i])))-self.random_start_range/2)
        return coefs

    def random_intercepts(self):
        layers = []
        for layer in self.hidden_layer_sizes:
            layers.append(layer)
        layers.append(1)

        intercepts = []
        for i in range(len(layers)):
            intercepts.append(self.random_start_range * np.random.random_sample(layers[i]))
        return intercepts

    def get_update(self, state):
        return self.predict(np.array(state).reshape(1, -1))[0]
