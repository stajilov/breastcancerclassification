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

from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.layers import Input, Conv2D, Lambda, merge, Flatten,MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K

import numpy.random as rng

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
   				output[col] = LabelEncoder().fit_transform(output[col])
   		else:
   			for colname,col in output.iteritems():
   				output[colname] = LabelEncoder().fit_transform(col)
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
        #tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        #filtered_tokens = []
        #for token in tokens:
        #    if re.search('[a-zA-Z]', token):
        #        filtered_tokens.append(token)
        #stems = [wordnet_lemmatizer.lemmatize(t) for t in filtered_tokens]
        #stems = [stemmer.stem(t) for t in filtered_tokens]
        
        players = text.split('|')
        
        return [x.lower() for x in players]


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



def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


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
train_test = TrainTestSplitter(target_column, train_test_ration)
print("Getting splits...")
X_train,X_test,y_train,y_test = train_test.execute(df)

x_train = X_train.as_matrix(columns=None)
x_test =  X_test.as_matrix(columns=None)
y_train = y_train.as_matrix(columns=None)
y_test = y_test.as_matrix(columns=None)


print("Starting model...")
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
#rmsprop

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

start = time()
model.fit(x_train, y_train,
          epochs=200,
          batch_size=64)
score = model.evaluate(x_test, y_test, batch_size=64)


print(model.metrics_names)
print("score: ", score)

print("Took seconds: ", str(time() - start))


np.savetxt("original.txt", y_test, newline=" ")

predictions = model.predict(x_test)

np.savetxt("predicted.txt", predictions, newline=" ")

rounded = [int(round(x[0])) for x in predictions]
np.savetxt("predicted_1.txt", rounded, newline=" ")


score = accuracy_score(y_test,rounded)*100
print(score, "%")






