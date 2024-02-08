import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV


# file paths
data_path = "reviews.csv"
train_path = "train.csv"
validation_path = "valid.csv"

# adjust these parameters to have different parts of the code run (default values display deliverables)
show_data_prep = True
show_hyperparam_tuning = False
show_final_model_training_testing = True

if show_data_prep:

    ### Most print statements have been commented out for a cleaner output

    # read in data
    data = pd.read_csv(data_path, delimiter='\t')

    # create sentiment column using mappings
    map_1 = {1:0, 2:0, 3:1, 4:2, 5:2}
    data['Sentiment'] = data['RatingValue'].map(map_1)

    # print(data['Sentiment'].value_counts())

    # now check how many positive and negative examples are in the data
    # print(data['Sentiment'].value_counts())
    positive_amount = data['Sentiment'].value_counts()[2]
    neutral_amount = data['Sentiment'].value_counts()[1]
    negative_amount = data['Sentiment'].value_counts()[0]
    # print(f"Amount of positive reviews: {positive_amount}, Amount of negative reviews: {negative_amount}, Amount of neutral reviews: {neutral_amount}")

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

    print("\nNormalized Dataset Value Counts:")
    print(data["Sentiment"].value_counts())

    # drop unnecessary columns
    data.drop(['RatingValue', 'DatePublished', 'Name'], axis=1, inplace=True)
    # data.info()

    # reset index
    data.reset_index(inplace=True, drop=True)

    # add number column
    data['Number'] = data.index
    # print("\nFinal DataFrame before train/validation split")
    # print(data.head())

    # train/test split (30% of data for testing)
    train, test = train_test_split(data, test_size=0.3)

    # save them as separate csvs
    train.to_csv('train.csv')
    test.to_csv('valid.csv')

if show_hyperparam_tuning:

    # build pipeline (for each model) to transform text from bag of words lecture
    text_clf_mnb = Pipeline([
        ('vectmnb', CountVectorizer()),
        ('tfidfmnb', TfidfTransformer()),
        ('clfmnb', MultinomialNB()),
    ])

    text_clf_sdg = Pipeline([
        ('vectsdg', CountVectorizer()),
        ('tfidfsdg', TfidfTransformer()),
        ('clfsdg', SGDClassifier(loss='hinge', penalty='l2',
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

    # print initial performance mnb
    print("Initial Performance Metrics of Multinomial Naieve Bayes (Before Hyperparameter Tuning)\n")

    # calculate overall accuracy and print it
    accuracy = metrics.accuracy_score(test['Sentiment'], pred_mnb)
    print(f"Accuracy Score: {accuracy}")

    # use classification report with output_dict=True to get a dictionary of class wise performance
    creport = metrics.classification_report(test['Sentiment'], pred_mnb, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
    print(f"Negative f1-score: {creport['Negative']['f1-score']}, Neutral f1-score: {creport['Neutral']['f1-score']}, Positive f1-score: {creport['Positive']['f1-score']}")

    # create confusion matrix and plot it
    cmatrix = metrics.confusion_matrix(test['Sentiment'], pred_mnb, labels=text_clf_mnb.classes_)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=['Negative', 'Neutral', 'Positive'])
    display.plot()
    plt.show()

    # print initial performance sdg
    print("Initial Performance Metrics of Support Vector Machine (Before Hyperparameter Tuning)\n")

    # calculate overall accuracy and print it
    accuracy = metrics.accuracy_score(test['Sentiment'], pred_sdg)
    print(f"Accuracy Score: {accuracy}")

    # use classification report with output_dict=True to get a dictionary of class wise performance
    creport = metrics.classification_report(test['Sentiment'], pred_sdg, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
    print(f"Negative f1-score: {creport['Negative']['f1-score']}, Neutral f1-score: {creport['Neutral']['f1-score']}, Positive f1-score: {creport['Positive']['f1-score']}")

    # create confusion matrix and plot it
    cmatrix = metrics.confusion_matrix(test['Sentiment'], pred_sdg, labels=text_clf_mnb.classes_)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=['Negative', 'Neutral', 'Positive'])
    display.plot()
    plt.show()

    # sdg is more promising, but hyper parameter tuning must be performed

    # hyper parameter tuning mnb

    # parameter dictionary
    params = {
        'vectmnb__ngram_range': [(1, 1), (1, 2)],
        'tfidfmnb__use_idf': (True, False),
        'clfmnb__alpha': (1e-2, 1e-3),
    }

    # create object
    gs_mnb = GridSearchCV(text_clf_mnb, params, cv=5, n_jobs=-1)

    # fit grid search
    gs_mnb.fit(train['Review'], train['Sentiment'])

    print(gs_mnb.best_params_)

    # hyper parameter tuning sdg

    # parameter dictionary
    params = {
        'vectsdg__ngram_range': [(1, 1), (1, 2)],
        'tfidfsdg__use_idf': (True, False),
        'clfsdg__alpha': (1e-2, 1e-3),
        'clfsdg__loss': ['hinge','log_loss','squared_hinge','squared_error','perceptron','huber'],
        'clfsdg__penalty': ['l2','l1',None],
        'clfsdg__max_iter': [5,50,100,500,1000,1500,2000],
        'clfsdg__learning_rate': ['constant','optimal','invscaling'],
    }

    # create object
    gs_sdg = GridSearchCV(text_clf_sdg, params, cv=5, n_jobs=-1)

    # fit grid search
    gs_sdg.fit(train['Review'], train['Sentiment'])

    print(gs_sdg.best_params_)

    # test models with best parameters

    # mnb
    text_clf_mnb = Pipeline([
        ('vectmnb', CountVectorizer(ngram_range=(1,2))),
        ('tfidfmnb', TfidfTransformer(use_idf=True)),
        ('clfmnb', MultinomialNB(alpha=0.01)),
    ])

    text_clf_sdg = Pipeline([
        ('vectsdg', CountVectorizer(ngram_range=(1,2))),
        ('tfidfsdg', TfidfTransformer(use_idf=True)),
        ('clfsdg', SGDClassifier(loss='squared_hinge', penalty='l2',
                            alpha=0.001, random_state=42,
                            max_iter=5, tol=None, learning_rate='optimal')),
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

    # print performance mnb
    print("Performance Metrics of Best Multinomial Naieve Bayes (After Hyperparameter Tuning)\n")

    # calculate overall accuracy and print it
    accuracy = metrics.accuracy_score(test['Sentiment'], pred_mnb)
    print(f"Accuracy Score: {accuracy}")

    # calcualte average f1-score
    f1_score = metrics.f1_score(test['Sentiment'], pred_mnb, average='macro')
    print(f"Average f1-score (macro averaged): {f1_score}")

    # use classification report with output_dict=True to get a dictionary of class wise performance
    creport = metrics.classification_report(test['Sentiment'], pred_mnb, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
    print(f"Negative f1-score: {creport['Negative']['f1-score']}, Neutral f1-score: {creport['Neutral']['f1-score']}, Positive f1-score: {creport['Positive']['f1-score']}")

    # create confusion matrix and plot it
    cmatrix = metrics.confusion_matrix(test['Sentiment'], pred_mnb, labels=text_clf_mnb.classes_)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=['Negative', 'Neutral', 'Positive'])
    display.plot()
    plt.show()

    # print performance sdg
    print("Performance Metrics of Best Support Vector Machine (After Hyperparameter Tuning)\n")

    # calculate overall accuracy and print it
    accuracy = metrics.accuracy_score(test['Sentiment'], pred_sdg)
    print(f"Accuracy Score: {accuracy}")

    # calcualte average f1-score
    f1_score = metrics.f1_score(test['Sentiment'], pred_sdg, average='macro')
    print(f"Average f1-score (macro averaged): {f1_score}")

    # use classification report with output_dict=True to get a dictionary of class wise performance
    creport = metrics.classification_report(test['Sentiment'], pred_sdg, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
    print(f"Negative f1-score: {creport['Negative']['f1-score']}, Neutral f1-score: {creport['Neutral']['f1-score']}, Positive f1-score: {creport['Positive']['f1-score']}")

    # create confusion matrix and plot it
    cmatrix = metrics.confusion_matrix(test['Sentiment'], pred_sdg, labels=text_clf_mnb.classes_)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=['Negative', 'Neutral', 'Positive'])
    display.plot()
    plt.show()

    # from performance metrics, sdg is the better model

if show_final_model_training_testing:

    # read in training set
    train = pd.read_csv(train_path)
    
    text_clf_sdg = Pipeline([
        ('vectsdg', CountVectorizer(ngram_range=(1,2))),
        ('tfidfsdg', TfidfTransformer(use_idf=True)),
        ('clfsdg', SGDClassifier(loss='squared_hinge', penalty='l2',
                            alpha=0.001, random_state=42,
                            max_iter=5, tol=None, learning_rate='optimal')),
    ])

    # fit and transform the data
    text_clf_sdg.fit(train['Review'], train['Sentiment'])

    # read in validation set
    test = pd.read_csv(validation_path)

    # test models
    pred_sdg = text_clf_sdg.predict(test['Review'])

    # print performance sdg
    print("\nPerformance Metrics of Best Support Vector Machine (After Hyperparameter Tuning)\n")

    # calculate overall accuracy and print it
    accuracy = metrics.accuracy_score(test['Sentiment'], pred_sdg)
    print(f"Accuracy Score: {accuracy}")

    # calcualte average f1-score
    f1_score = metrics.f1_score(test['Sentiment'], pred_sdg, average='macro')
    print(f"Average f1-score (macro averaged): {f1_score}")

    # use classification report with output_dict=True to get a dictionary of class wise performance
    creport = metrics.classification_report(test['Sentiment'], pred_sdg, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
    print(f"Negative f1-score: {creport['Negative']['f1-score']}, Neutral f1-score: {creport['Neutral']['f1-score']}, Positive f1-score: {creport['Positive']['f1-score']}")

    # create confusion matrix and plot it
    cmatrix = metrics.confusion_matrix(test['Sentiment'], pred_sdg, labels=text_clf_mnb.classes_)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=['Negative', 'Neutral', 'Positive'])
    display.plot()
    plt.show()
