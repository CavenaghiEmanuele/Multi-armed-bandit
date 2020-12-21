from numpy.core.defchararray import count
import pandas as pd
import numpy as np


if __name__ == '__main__':

    dataset = pd.read_csv('adige_news.csv')
    article_topic = pd.read_csv('article_topic_category.csv')

    dataset['article_topic'] = pd.Series(np.zeros(len(dataset.index)), index=dataset.index)
    for i in range(len(dataset.index)):
        try:
            id_row = article_topic.loc[article_topic['article_id'] == dataset.iloc[i]['pk_article']]['percentage'].idxmax()
            dataset.loc[i, 'article_topic'] = article_topic.iloc[id_row]['topic_id']
        except:
            pass

    dataset.drop(['pk_user', 'pk_session', 'timeview', 'date-time'],
                 axis=1,
                 inplace=True)
    dataset.to_csv('modified_adige_news.csv', index=False)
