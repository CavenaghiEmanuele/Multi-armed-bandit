import gzip
import re
import csv


if __name__ == "__main__":

    path = "/media/emanuele/860EFA500EFA392F/Dataset Yahoo!/R6/"

    filenames = [
        "ydata-fp-td-clicks-v1_0.20090501",
        "ydata-fp-td-clicks-v1_0.20090502",
        "ydata-fp-td-clicks-v1_0.20090503",
        "ydata-fp-td-clicks-v1_0.20090504",
        "ydata-fp-td-clicks-v1_0.20090505",
        "ydata-fp-td-clicks-v1_0.20090506",
        "ydata-fp-td-clicks-v1_0.20090507",
        "ydata-fp-td-clicks-v1_0.20090508",
        "ydata-fp-td-clicks-v1_0.20090509",
        "ydata-fp-td-clicks-v1_0.20090510",
    ]

    for filename in filenames:
        file_path = path + filename + ".gz"
        input = gzip.GzipFile(file_path, 'rb')
        data = input.read().decode("utf-8")
        input.close()
        print("Load file: " + filename)

        result = re.findall("(\d* \d* \d)|\r", data)
        print("Cut file: " + filename)

        save_file_path = path + "cut_dataset/" + filename + ".csv"
        with open(save_file_path, 'w', newline="\n") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, delimiter='\n')
            wr.writerow(result)
