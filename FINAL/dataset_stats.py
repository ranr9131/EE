import pandas as pd


CSV_COLUMN_NAMES = ['num']
tempString = ""
for i in range(25):
    for j in range(25):
        tempString = f"{i}.{j}"
        CSV_COLUMN_NAMES.append(tempString)
NUMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


train = pd.read_csv(r"data/numbers_training.csv", names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(r"data/numbers_test.csv", names=CSV_COLUMN_NAMES, header=0)


train_y = train.pop("num")
test_y = test.pop("num")




#TRAINING SET
all_nums_training = list(train_y)
num_distribution_training = [0 for i in range(10)]
for i in range(len(all_nums_training)):
    num_distribution_training[all_nums_training[i]] = num_distribution_training[all_nums_training[i]] + 1

print("TRAINING SET:")
print(f"Samples: {sum(num_distribution_training)}")
for i in range(10):
    print(f"{i}: {(num_distribution_training[i]/sum(num_distribution_training))*100}%, samples: {num_distribution_training[i]}")


print()

#TESTING SET
all_nums_testing = list(test_y)
num_distribution_testing = [0 for i in range(10)]
for i in range(len(all_nums_testing)):
    num_distribution_testing[all_nums_testing[i]] = num_distribution_testing[all_nums_testing[i]] + 1

print("TESTING SET:")
print(f"Samples: {sum(num_distribution_testing)}")
for i in range(10):
    print(f"{i}: {(num_distribution_testing[i]/sum(num_distribution_testing))*100}%, samples: {num_distribution_testing[i]}")
