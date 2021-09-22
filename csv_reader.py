import csv
import sys

def CSVReader(filename):

    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:

        csvreader = csv.reader(csvfile)

        for row in csvreader:
            rows.append(row)

        line_num = csvreader.line_num

        return rows, line_num


def CSVWriter(filename, excel):
    with open(filename, 'w') as output_file:
        wr = csv.writer(output_file, 'excel')

        for each in excel:
            wr.writerow([str(each)])

