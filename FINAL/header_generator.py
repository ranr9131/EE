import csv

with open(r'numbers_training.csv', mode='a') as csv_file:
    fieldnames = [0, 625]
    tempString = ""
    for i in range(25):
        for j in range(25):
            tempString = f"{i}.{j}"
            fieldnames.append(tempString)


    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    # writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
    # writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})