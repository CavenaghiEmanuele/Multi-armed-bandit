import gzip
import re
import csv
import pickle
import pprint


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
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
    
    for i in range(1, 11):
        # Get id_articles
        path = '/home/emanuele/GoogleDrive/Thompson Sampling/yahoo_dataset/'
        with open(path + 'day' + str(i) + '/id_articles.txt', "rb") as fp:
            id_articles = pickle.load(fp)
        
        path = "/media/emanuele/860EFA500EFA392F/Dataset Yahoo!/R6/"
        
        file_path = path + filenames[i-1] + ".gz"
        input = gzip.GzipFile(file_path, 'rb')
        data = input.read().decode("utf-8")
        input.close()
        print("Load file: " + filenames[i-1])

        features = {}
        for id in id_articles:
            s = '\|' + str(id) + ' \d\:(.{8}) \d\:(.{8}) \d\:(.{8}) \d\:(.{8}) \d\:(.{8}) \d\:.{8}'
            result = re.search(s, data)
            if result != None:
                features.update({id : result.groups()})
            else:
                features.update({id : result})

        print("Cut file: " + filenames[i-1])
        
        path = '/home/emanuele/GoogleDrive/Thompson Sampling/yahoo_dataset/'
        save_file_path = path + 'day' + str(i) + '.csv'
        with open(save_file_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['id_article', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
            for key, value in features.items():
                l = [key]
                if value != None:
                    l.extend([item for item in value])
                else:
                    l.append(None)
                writer.writerow(l)
