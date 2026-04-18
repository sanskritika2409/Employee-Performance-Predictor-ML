from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    le = LabelEncoder()

    data['department'] = le.fit_transform(data['department'])
    data['performance'] = le.fit_transform(data['performance'])

    X = data.drop('performance', axis=1)
    y = data['performance']

    return X, y