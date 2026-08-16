import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

data = {
    'Name': ['Anna', 'Bob', 'Charlie', 'Diana', 'Eric'],
    'Age': [20, 34, 23, None, 33],
    'Gender': ['f', 'm', 'm', 'f', 'm'],
    'Job' : ['Programmer', 'Writter', 'Cook', 'Programmer', 'Teacher']
}

df = pd.DataFrame(data)

# Preprocessing Pipelines
    # Drop Name Feature
    # Impute Ages
    # Turn Gender into Binary / Numeric
    # One-Hot Encode Jobs
    
# Drop Name Feature
df = df.drop(['Name'], axis=1)

# Impute Ages
imputer = SimpleImputer(strategy="mean")
df['Age'] = imputer.fit_transform(df[['Age']])

# Numeric Gender
gender_dct = {'m' : 0, 'f' : 1}
df['Gender'] = [gender_dct[g] for g in df['Gender']]

# OneHotEncode Job
encode = OneHotEncoder()
matrix = encode.fit_transform(df[['Job']]).toarray()

column_names = ['Programmer', 'Writter', 'Cook', 'Teacher']

for i in range(len(matrix.T)):
    df[column_names[i]] = matrix.T[i]

df = df.drop(['Job'], axis=1)

# The use of Pipelines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class NameDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(['Name'], axis=1)
class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
            return self
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X
class GenderNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
            return self
    def transform(self, X):
        gender_dct = {'m' : 0, 'f' : 1}
        X['Gender'] = [gender_dct[g] for g in X['Gender']]
        return X         
class OneHotEncode(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
            return self
    def transform(self, X):
        encode = OneHotEncoder()
        matrix = encode.fit_transform(X[['Job']]).toarray()
        
        column_names = ['Programmer', 'Writter', 'Cook', 'Teacher']

        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        X = X.drop(['Job'], axis=1)
        return X
  
          
data_2 = {
    'Name': ['Fiona', 'Gerald', 'Hans', 'Isabella', 'Jacob'],
    'Age': [20, 34, None, None, 33],
    'Gender': ['f', 'm', 'm', 'f', 'm'],
    'Job' : ['Writter', 'Programmer', 'Programmer', 'Programmer', 'Teacher']
}

df2 = pd.DataFrame(data_2)


# dropper = NameDropper()
# imputer = AgeImputer()
# genderNumeric = GenderNumeric()
# job = OneHotEncode()


# df2 = dropper.fit_transform(df2)
# df2 = imputer.fit_transform(dropper.fit_transform(df2))
# df2 = genderNumeric.fit_transform(imputer.fit_transform(dropper.fit_transform(df2)))


# df2 = job.fit_transform(genderNumeric.fit_transform(imputer.fit_transform(dropper.fit_transform(df2))))
pipe = Pipeline([
    ('dropper', NameDropper()),
    ('imputer', AgeImputer()),
    ('genderNumeric', GenderNumeric()),
    ('job', OneHotEncode()),
])
pipe = pipe.fit_transform(df2)
print(pipe)