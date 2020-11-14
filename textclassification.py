import sys
import nltk
import sklearn
import pandas as pd
import numpy as np

# print('Python: {}'.format(sys.version))
# print('NLTK: {}'.format(nltk.__version__))
# print('Scikit-learn: {}'.format(sklearn.__version__))
# print('Pandas: {}'.format(pandas.__version__))
# print('NumPy: {}'.format(numpy.__version__))

# Load the dataset


df = pd.read_table('SMSSpamCollection', header=None, encoding='utf-8')

# Print useful information about the dataset

# print(df.info())
# print(df.head())

# check class distribution
classes = df[0]
# print(classes.value_counts())

# Preprocess the Data
# convert class labels to binary values, 0 = ham, 1 = spam

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

# print(classes[:10])
# print(Y[:10])

# store the SMS message data
text_messages = df[1]
# print(text_messages[:10])

#  ** Regular Expressions
# Some common regular expression metacharacters - copied from wikipedia

# ^ Matches the starting position within the string. In line-based tools, it matches the starting position of any line.

# . Matches any single character (many applications exclude newlines, and exactly which characters are considered newlines is flavor-, character-encoding-, and platform-specific, but it is safe to assume that the line feed character is included). Within POSIX bracket expressions, the dot character matches a literal dot. For example, a.c matches "abc", etc., but [a.c] matches only "a", ".", or "c".

# [ ] A bracket expression. Matches a single character that is contained within the brackets. For example, [abc] matches "a", "b", or "c". [a-z] specifies a range which matches any lowercase letter from "a" to "z". These forms can be mixed: [abcx-z] matches "a", "b", "c", "x", "y", or "z", as does [a-cx-z]. The - character is treated as a literal character if it is the last or the first (after the ^, if present) character within the brackets: [abc-], [-abc]. Note that backslash escapes are not allowed. The ] character can be included in a bracket expression if it is the first (after the ^) character: []abc].

# [^ ] Matches a single character that is not contained within the brackets. For example, [^abc] matches any character other than "a", "b", or "c". [^a-z] matches any single character that is not a lowercase letter from "a" to "z". Likewise, literal characters and ranges can be mixed.

# $ Matches the ending position of the string or the position just before a string-ending newline. In line-based tools, it matches the ending position of any line.

# ( ) Defines a marked subexpression. The string matched within the parentheses can be recalled later (see the next entry, \n). A marked subexpression is also called a block or capturing group. BRE mode requires ( ).

# \n Matches what the nth marked subexpression matched, where n is a digit from 1 to 9. This construct is vaguely defined in the POSIX.2 standard. Some tools allow referencing more than nine capturing groups.

# * Matches the preceding element zero or more times. For example, abc matches "ac", "abc", "abbbc", etc. [xyz] matches "", "x", "y", "z", "zx", "zyx", "xyzzy", and so on. (ab)* matches "", "ab", "abab", "ababab", and so on.

# {m,n} Matches the preceding element at least m and not more than n times. For example, a{3,5} matches only "aaa", "aaaa", and "aaaaa". This is not found in a few older instances of regexes. BRE mode requires {m,n}.

# use regular expressions to replace email address, urls, phone numbers, other numbers, symbols

# replace email adresses with 'emailaddr'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')

# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')

# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')


# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

# change words to lower case - Hello, HELLO, hello are all the same word!
processed = processed.str.lower()
# print(processed)

# remove stop words from text messages

from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: " ".join(term for term in x.split() if term not in stop_words))

# remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: " ".join(ps.stem(term) for term in x.split()))

# print(processed)

from nltk.tokenize import word_tokenize

# creating a bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

# print the total number of words and 15 most common words
# print('Number of words : {}'.format(len(all_words)))
# print('Most common words: {}'.format(all_words.most_common(15)))

# use the 1500 most common words as features

word_features = list(all_words.keys())[:1500]

# define a find_features


def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


# Lets see an example
# features = find_features(processed[0])
# for key, value in features.items():
#     if value == True:
#         print(key)

# find features for all messages
# for python 3+ use list(zip(contents))
messages = list(zip(processed, Y))

# define a seed for reproducibilty
seed = 1
np.random.seed = seed

np.random.shuffle(messages)

# call find_features function for each SMS messages
featuresets = [(find_features(text), label) for (text, label) in messages]

# split training and testing data sets using sklearn
from sklearn import model_selection

training, testing = model_selection.train_test_split(featuresets, test_size=0.25, random_state=seed)

# print("Training: {}".format(len(training)))
# print("Testing: {}".format(len(testing)))

# * Scikit-Learn classifiers with NLTK

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ['N Nearest Neighrbors', "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier", "Naive Bayes", "SVM Linear"]
classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
               LogisticRegression(),
               SGDClassifier(max_iter=100),
               MultinomialNB(),
               SVC(kernel='linear')]

models = zip(names, classifiers)

# print(models)

# wrap models in NLTK

from nltk.classify.scikitlearn import SklearnClassifier

# for name, model in models:
#     nltk_model = SklearnClassifier(model)
#     nltk_model.train(training)
#     accuracy = nltk.classify.accuracy(nltk_model, testing) * 100
# print('{}: Accuracy: {}'.format(name, accuracy))

# Ensemble method - Voting classifier
from sklearn.ensemble import VotingClassifier

# Define models to train

names = ['N Nearest Neighrbors', "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier", "Naive Bayes", "SVM Linear"]

classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
               LogisticRegression(),
               SGDClassifier(max_iter=100),
               MultinomialNB(),
               SVC(kernel='linear')]

models = list(zip(names, classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models, voting='hard', n_jobs=-1))
nltk_ensemble.train(training)
# accuracy = nltk.classify.accuracy(nltk_ensemble, testing) * 100
# print('Ensemble Method Accuracy: {}'.format(accuracy))

# make class label prediction for testing set

txt_features, labels = list(zip(*testing))

prediction = nltk_ensemble.classify_many(txt_features)

# print a confusion matrix and a classification report
print(classification_report(labels, prediction))


pd.DataFrame(
    confusion_matrix(labels, prediction),
    index=[['actual', 'actual'], ['ham', 'spam']],
    columns=[['predicted', 'predicted'], ['ham', 'spam']])
