import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.metrics = {}

    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
            else:
                self.metrics[key] = [value]

        self.plot()

    def plot(self):
        self.ax.clear()
        for key, values in self.metrics.items():
            self.ax.plot(values, label=key)
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Metric')
        self.ax.legend(loc='upper left')
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)