import pandas as pd


def create_data_set(file_name, target_name):
    data_set = pd.read_csv(file_name, header=None)
    data_set = data_set.dropna()
    labels = data_set[0]
    text = data_set[1] + data_set[2]
    new_data_set = pd.DataFrame({'labels': labels, 'text':text})
    new_data_set.to_csv(target_name, index=False)

create_data_set('data/ag_news_csv/train.csv', 'data/train.csv')
create_data_set('data/ag_news_csv/train.csv', 'data/test.csv')
