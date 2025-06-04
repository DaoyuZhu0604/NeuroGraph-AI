import matplotlib.pyplot as plt

def plot_results(metrics):
    for key, values in metrics.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.show()
