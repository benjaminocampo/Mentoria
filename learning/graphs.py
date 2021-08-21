import matplotlib.pyplot as plt


# TODO: Improve these plots. Maybe we can use just one containing the metrics we
# want
def plot_accuracy_curve(records):
    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    fig.title("accuracy")
    fig.plot(records)
    fig.xlabel("epoch")
    fig.ylabel("accuracy")
    fig.legend(["train"], loc="upper left")
    return fig