import matplotlib.pyplot as plt

def plot_heatmap(data, experiment, epoch):
    plt.clf()
    heatmap = plt.imshow(data.T, cmap='viridis')

    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #        'dog', 'frog', 'horse', 'ship', 'truck']

    # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    classes = ["Switchgrass", "Cotton", "Peanut", "Sesame", "Sunflower", "Papaya"]

    plt.xlabel("True label")
    plt.ylabel("Predicted Label")
    # plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels=classes)
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=classes)
    # plt.yticks(ticks=[0, 1], labels=["Not Three", "Three"])
    plt.yticks(ticks=[0, 1], labels=["Switchgrass", "Not Switchgrass"])
    plt.title('Heatmap')

    for i in range(data.T.shape[0]):
        for j in range(data.T.shape[1]):
            plt.text(j, i, f"{data[j, i]:.0f}", ha='center', va='center', color='w')

    experiment.log_figure(figure_name="heatmap", figure=heatmap.figure, step=epoch)