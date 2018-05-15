import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score


from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#from keras.models import Sequential
#from keras.layers import Dense, Dropout

#from keras.layers import Input, Conv2D, Lambda, merge, Flatten,MaxPooling2D
#from keras.models import Model
#from keras.regularizers import l2
#from keras import backend as K

import numpy.random as rng
from sklearn.externals import joblib


model_file_name = 'breast_prediction.pkl'
labels_file_name = 'labels.pkl'



#splitter
class TrainTestSplitter():
    def __init__(self, column, ration):
        self.trainTestRatio = ration
        self.targetColumn = column

    def execute(self, df):
        y = df.pop(self.targetColumn)
        X = df
        X_tr,X_test,y_train,y_test = train_test_split(X.index,y,test_size=self.trainTestRatio)
        df_train = X.loc[X_tr]
        df_test = X.loc[X_test]		
        return df_train,df_test,y_train,y_test

class TrainTestSplitterFull():
    def __init__(self, column, ration):
        self.trainTestRatio = ration
        self.targetColumn = column

    def execute(self, df):
        y = df.pop(self.targetColumn)
        X = df
        return X,y

class ColumnsEncoder():
    def __init__(self):
        self.columns = []

    def execute(self, df, columns):
        encoded = self.transform(df, columns)
        return encoded


    def transform(self,X,columns):
        output = X.copy()
        if columns is not None:
            for col in columns:
                le = LabelEncoder()
                output[col] = le.fit_transform(output[col])
                le_name_mapping = dict( zip(le.transform(le.classes_), le.classes_  ) )

                joblib.dump(le_name_mapping, labels_file_name)
                print(le_name_mapping)
                
        else:
            for colname,col in output.iteritems():
                le = LabelEncoder()
                output[colname] = le.fit_transform(col)
                le_name_mapping = dict( zip(le.transform(le.classes_), le.classes_  ) )

                joblib.dump(le_name_mapping, labels_file_name)
                
                print(le_name_mapping)
                
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


class ColumnsRemover():

    def __init__(self):
        self.columns = []

    def execute(self, df, columns):
        for c in columns:
            df.drop(c, axis=1, inplace=True)
        return df

class ColumnsFilter():
    def __init__(self):
        self.columns = []

    def execute(self, df, columns):
        for c in columns:
            df = df[df[c].notnull()]
        return df


class TfIdfProcessor():

    def __init__(self):
        self.columns = []

    def tokenize_and_stem(self,text):
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [wordnet_lemmatizer.lemmatize(t) for t in filtered_tokens]
        stems = [stemmer.stem(t) for t in filtered_tokens]
        
        #players = text.split('|')
        
        return [x.lower() for x in stems]


    def tokenize_only(text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
           if re.search('[a-zA-Z]', token):
               filtered_tokens.append(token)
        return filtered_tokens

    def getTfIdfMatrixForDF(self, df,columns):
        local_df = df
        tfidf_vectorizer = CountVectorizer(tokenizer=self.tokenize_and_stem, binary=True)
        for c in columns:
            #print(c)
            valuesOfDF = local_df.pop(c).values
            #print(valuesOfDF)
            X = tfidf_vectorizer.fit_transform(valuesOfDF.astype('U')).toarray()
            for i, col in enumerate(tfidf_vectorizer.get_feature_names()):
                local_df[col] = X[:, i]

        return local_df

    def execute(self, df, columns):
        transformed = self.getTfIdfMatrixForDF(df,columns)
        return transformed




df = pd.read_csv('data.csv', sep=',')

columns_to_remove = ['id']
remover = ColumnsRemover()
df = remover.execute(df, columns_to_remove)



columns_to_filter_none = []
filt = ColumnsFilter()
df = filt.execute(df, columns_to_filter_none)


player_columns = []
tf = TfIdfProcessor()
df = tf.execute(df, player_columns)

#df.to_csv('test_player_matrix_1.csv')


columns_to_encode = ['diagnosis']
enc = ColumnsEncoder()
df = enc.execute(df, columns_to_encode)



print(df.head(5))
print(df.shape)


target_column = 'diagnosis'
train_test_ration = 0.2


#the one used for 20% train test split
cut_df = df.copy()

train_test = TrainTestSplitter(target_column, train_test_ration)
print("Getting splits...")
X_train,X_test,y_train,y_test = train_test.execute(cut_df)

x_train = X_train.as_matrix(columns=None)
x_test =  X_test.as_matrix(columns=None)
y_train = y_train.as_matrix(columns=None)
y_test = y_test.as_matrix(columns=None)


#the one used for 100% train
full_df = df.copy()
train_test_full = TrainTestSplitterFull(target_column, train_test_ration)
full_X,full_Y  = train_test_full.execute(full_df)


classifiers = {
"gbr": GradientBoostingClassifier(),
"logistic" : LogisticRegression(penalty='l1'),
"kneighbours" : KNeighborsClassifier(n_neighbors=5),
"rft" : RandomForestClassifier(n_estimators=10),
"dtr": DecisionTreeClassifier(max_depth=5),
"MLP": MLPClassifier(),
"MltnmnlNB": MultinomialNB(),

 "etc":ExtraTreesClassifier(),
 "adc" : AdaBoostClassifier(),
  "sgdc" :SGDClassifier(),
  "svm": SVC()
}


#ALL CLASSIFIERS
results = {}

for name, clf in classifiers.items():

	nm = str(clf.__class__.__name__)
	print("\nPredicting with %s" % nm)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)
	score = accuracy_score(y_test,predictions)*100
	print(score, "%")
	results[nm] = score
	#sliced_df[nm] = predictions
	
	
#sliced_df.to_csv('original_with_predicted')
print("\nResults:")
print(results)

best_clf = classifiers['adc']
best_clf.fit(full_X, full_Y)
new_value = np.array([10.05,21.38,122.8,1200,0.1184,0.2776,0.3001,0.2011,0.2001,0.0893,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]).reshape(1, -1)
predicted = best_clf.predict(new_value)
print(predicted)

joblib.dump(best_clf, model_file_name)








