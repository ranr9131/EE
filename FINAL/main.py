import tensorflow_model
import matplotlib.pyplot as plt
import numpy as np
import os
import ast


def get_plot(training_dir: str, test_dir: str, epoch_num: int, batch_size: int, output_dir: str):
    train, test = tensorflow_model.load_data(training_dir, test_dir)
    train, train_y, test, test_y = tensorflow_model.randomize(train, test)

    model = tensorflow_model.initiate_model()

    log = tensorflow_model.run_model(model = model, train = train, train_y = train_y, test = test, test_y = test_y, epoch_num = epoch_num, batch_size = batch_size)


    # Plot
    x_epochs = np.array(list(log.keys()))
    y_training_acc = np.array([log[i]["training_accuracy"] for i in range(1, epoch_num+1)])
    plt.plot(x_epochs, y_training_acc)

    y_test_acc = np.array([log[i]["test_accuracy"] for i in range(1, epoch_num+1)])
    plt.plot(x_epochs, y_test_acc)


    plt.xlabel("Time(Epoch)")
    plt.ylabel("Accuracy(%)")
    plt.title("Model Accuracy")

    plt.savefig(f"{output_dir}/epoch{str(epoch_num)}-batch{str(batch_size)}.png")
    plt.clf()

    return log

def write_log_txt(log, output_path: str):
    with open(output_path, "w") as log_file:
        for i in range(1, len(log)+1):
            log_file.write(f'epoch: {i}, training_loss: {log[i]["training_loss"]}, training_accuracy: {log[i]["training_accuracy"]}, test_loss: {log[i]["test_loss"]}, test_accuracy: {log[i]["test_accuracy"]}\n')


def get_combined_plot(epoch_dir: str):
    filenames = []
    for filename in os.listdir(epoch_dir):
        filenames.append(filename)
    
    # Sort by batch size
    filenames.sort(key = lambda x: int(x.split("batch")[1]))

    for filename in filenames:
        with open(f"{epoch_dir}/{filename}/{filename}.txt", "r") as log:
            lines = log.readlines()
            lines = list(map(lambda x: x[:-1], lines))
            lines = list(map(lambda x: x.split(", "), lines))
            lines = list(map(lambda x: list(map(lambda y: y.split(": "), x)), lines))

            log_dict = {}
            for line in lines:
                temp_dict = {}
                for feat in line[1:]:
                    temp_dict[feat[0]] = float(feat[1])
                log_dict[int(line[0][1])] = temp_dict


        # Graph batch line
        num_epochs = list(log_dict.keys())[-1]
        x_epochs = np.array(list(log_dict.keys()))
        y_test_acc = np.array([log_dict[i]["test_loss"] for i in range(1, num_epochs+1)])

        plt.plot(x_epochs, y_test_acc)

    # Labels
    plt.xlabel("Time(Epoch)")
    plt.ylabel("Loss")
    plt.title(f"Test Loss Epoch {epoch_dir.split('h')[1]}")

    # Legend
    filenames = list(map(lambda x: x.split("batch"), filenames))
    filenames = list(map(lambda x: int(x[1]), filenames))
    plt.legend(filenames, title="Batch Size", loc="upper right")


    plt.savefig(f"combined_plots_test_loss/{epoch_dir.split('/')[1]}.png")

    plt.clf()




training_dir = r"data/numbers_training.csv"
test_dir = r"data/numbers_test.csv"

epochs = [10, 15, 20, 25, 30, 35, 40, 45, 50]
batch_sizes = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1500]

epochs = [500, 2000]

for epoch_num in epochs:
    epoch_dir = f"plots/epoch{str(epoch_num)}"
    os.mkdir(epoch_dir)
    for batch_size in batch_sizes:
        epoch_batch_dir = f"{epoch_dir}/epoch{str(epoch_num)}-batch{str(batch_size)}"
        os.mkdir(epoch_batch_dir)

        log = get_plot(training_dir, test_dir, epoch_num, batch_size, epoch_batch_dir)
        write_log_txt(log, f"{epoch_batch_dir}/epoch{str(epoch_num)}-batch{str(batch_size)}.txt")



# for filename in os.listdir("plots"):
#     get_combined_plot(f"plots/{filename}")
