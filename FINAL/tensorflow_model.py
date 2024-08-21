import tensorflow as tf
import pandas as pd
import numpy
import random


def load_data(training_csv_dir: str, test_csv_dir: str):
    CSV_COLUMN_NAMES = ['num']
    tempString = ""
    for i in range(25):
        for j in range(25):
            tempString = f"{i}.{j}"
            CSV_COLUMN_NAMES.append(tempString)
    NUMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


    train = pd.read_csv(training_csv_dir, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_csv_dir, names=CSV_COLUMN_NAMES, header=0)

    return train, test


def randomize(train, test, seed=None):
    if seed == None:
        train = train.sample(frac=1)
        test = test.sample(frac=1)
    else:
        train = train.sample(frac=1, random_state = seed)
        test = test.sample(frac=1, random_state = seed)

    train_y = train.pop("num")
    test_y = test.pop("num")

    return train, train_y, test, test_y


def initiate_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def run_model(model, train, train_y, test, test_y, epoch_num: int, batch_size: int):
    log = {}

    for i in range(epoch_num):
        print(f"Epoch: {i+1}")
        history = model.fit(train, train_y, epochs=1, batch_size=batch_size)

        #eval
        loss, acc = model.evaluate(test, test_y)

        temp_log = {}
        temp_log["training_loss"] = history.history["loss"][0]
        temp_log["training_accuracy"] = history.history["accuracy"][0]
        temp_log["test_loss"] = loss
        temp_log["test_accuracy"] = acc

        log[i+1] = temp_log
        print(temp_log)
        print()

    return log


if __name__ == "__main__":
    training_dir = r"data/numbers_training.csv"
    test_dir = r"data/numbers_test.csv"
    train, test = load_data(training_dir, test_dir)
    train, train_y, test, test_y = randomize(train, test)

    print(train)
    print(train_y)
    print(test)
    print(test_y)


    # model = initiate_model()

    # run_model(model = model, train = train, train_y = train_y, test = test, test_y = test_y, epoch_num = 30, batch_size = 32)


# predict = model.predict([test])
# for i in range(len(predict)):
#     print(numpy.argmax(predict[i]))