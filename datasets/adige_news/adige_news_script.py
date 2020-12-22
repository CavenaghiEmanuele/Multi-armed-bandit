from numpy.core.defchararray import count
import pandas as pd
import numpy as np


if __name__ == '__main__':

    dataset = pd.read_csv('original_adige_news.csv')
    article_topic = pd.read_csv('article_topic_category.csv')

    dataset['article_topic'] = pd.Series(np.zeros(len(dataset.index)), index=dataset.index)
    for i in range(len(dataset.index)):
        try:
            id_topics = article_topic.loc[article_topic['article_id'] == dataset.iloc[i]['pk_article']]['topic_id']
            if len(id_topics) > 0: # Not all articles have a topic associated
                dataset.loc[i, 'article_topic'] = ','.join(str(x) for x in id_topics)
            else: raise Exception
        except:
            dataset.drop(i, inplace=True)

    dataset.drop(['pk_user', 'pk_session', 'timeview', 'date-time'],
                 axis=1,
                 inplace=True)
    dataset.to_csv('adige_news.csv', index=False)
