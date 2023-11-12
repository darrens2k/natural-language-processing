import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# file paths
data_path = ""
train_path = ""
validation_path = ""


# read in data
data = pd.read_csv('reviews.csv', delimiter='\t')

# create sentiment column using mappings
map_1 = {1:0, 2:0, 3:1, 4:2, 5:2}
data['Sentiment'] = data['RatingValue'].map(map_1)

data['Sentiment'].value_counts()

# now check how many positive and negative examples are in the data
print(data['Sentiment'].value_counts())
positive_amount = data['Sentiment'].value_counts()[2]
neutral_amount = data['Sentiment'].value_counts()[1]
negative_amount = data['Sentiment'].value_counts()[0]
print(f"Amount of positive reviews: {positive_amount}, Amount of negative reviews: {negative_amount}, Amount of neutral reviews: {neutral_amount}")

# drop positives until it equals the average of the neutral and negative reviews
average_neutral_negative = (neutral_amount + negative_amount) / 2

# grab the indices of all positive reviews
positive_indices = data[data['Sentiment'] == 2].index
# convert to a numpy array
positive_indices = np.array(positive_indices)
# randomly shuffle the indices
np.random.shuffle(positive_indices)

# loop counter
counter = 0

# delete positive reviews at random until amount is reduced to desired value
while positive_amount > average_neutral_negative:

    # drop record
    data.drop(index=positive_indices[counter], inplace=True)

    # update amount of positive reviews
    positive_amount = data['Sentiment'].value_counts()[2]

    # increment counter
    counter += 1

print("\nNormalized Dataset")
print(data["Sentiment"].value_counts())

# drop unnecessary columns
data.drop(['RatingValue', 'DatePublished', 'Name'], axis=1, inplace=True)
data.info()

# reset index
data.reset_index(inplace=True, drop=True)

# add number column
data['Number'] = data.index
print("\nFinal DataFrame before train/validation split")
print(data.head())

# train/test split (30% of data for testing)
train, test = train_test_split(data, test_size=0.3)

# save them as separate csvs
train.to_csv('train.csv')
test.to_csv('valid.csv')

# build pipeline (for each model) to transform text from bag of words lecture
text_clf_mnb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf_sdg = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

# read in training csv
train = pd.read_csv('train.csv')

# fit and transform the data
text_clf_mnb.fit(train['Review'], train['Sentiment'])
text_clf_sdg.fit(train['Review'], train['Sentiment'])

# read in validation set
test = pd.read_csv('valid.csv')

# test models in their current state and print performance metrics before hyperparameter tuning
pred_mnb = text_clf_mnb.predict(test['Review'])
pred_sdg = text_clf_sdg.predict(test['Review'])

# print initial performance
print("\nInitial Performance (before hyperparameter tuning)\n")
print("Initial Metrics of Multinomial Naieve Bayes\n")
metrics.confusion_matrix(test['Sentiment'], pred_mnb)
print(metrics.classification_report(test['Sentiment'], pred_mnb))


