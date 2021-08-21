import matplotlib.pyplot as plt


# TODO: Improve these plots. Maybe we can use just one containing the metrics we
# want
def plot_accuracy_curve(records):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    ax.set_title("accuracy")
    ax.plot(records)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend(["train"], loc="upper left")
    return fig