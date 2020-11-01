import random
import nltk
from nltk.corpus import movie_reviews
import sklearn
from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

# build list of documents
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# shuffle the documents
random.shuffle(documents)

# print('Number of Documents:{}'.format(len(documents)))
# print('First Review:{}'.format(documents[0]))

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# print('Most common words: {}'.format(all_words.most_common(15)))
# print('The word happy: {}'.format(all_words['happy']))
# print(len(all_words))
# We'll use the 4000 most common words as features
word_features = list(all_words.keys())[:4000]

# Build a find_features function that will determine which of the 4000 word features are contained in a review


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# Lets use an example from a negative review
# features = find_features(movie_reviews.words('neg/cv000_29416.txt'))
# for key, value in features.items():
#     if value == True:
#         print(key)

# print(features)

# now lets do it for all the documents
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# we can split the featuresets into training and testing datasets using sklearn


# define a seed for reproductibility
seed = 1

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size=0.25, random_state=seed)

# print(len(training))
# print(len(testing))



# How we use sklearn algorithms in NLTK
model = SklearnClassifier(SVC(kernel='linear'))

# train the model on the training data
model.train(training)

# test on the testing dataset
accuracy = nltk.classify.accuracy(model, testing)
print('SVC Accuracy: {}'.format(accuracy))
