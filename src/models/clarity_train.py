import argparse # parse arguments as a dictionary for easy access
import gzip # to open gzip compressed files
import logging # to track and log intermediate steps as well as errors
import pickle # serialize python object(vector and classifier)
import time # to keep track of time taken to run vertain segments of code
import numpy as np

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# util classes
from features import preprocessor as pp
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='Perform clarity probability score estimation with the following arguments')
parser.add_argument('--x_train', type=str, required=True, help='Location of the training data')
parser.add_argument('--y_train', type=str, required=True, help='Location of the training labels')

parser.add_argument('--x_test', type=str, required=False, help='Location of the test/validation data')
# parser.add_argument('--y_test', type=str, required=False, help='Location of the test/validation labels')

parser.add_argument('--pp', type=bool, default=False, help='Preprocess data or not')
# other features

parser.add_argument('--model', type=str, required='nb', choices=['nb', 'lr', 'svc'], help='Name and directory to store trained model')
parser.add_argument('--feat', type=str, default='tf', choices=['tf', 'tfidf', 'dict'], help='Type of features to use')

parser.add_argument('--results', type=str, default=None, help='Location and filename of results output file')


args = parser.parse_args()
cmd_args = vars(args)

preprocess = cmd_args['pp']

def run():

    for key in cmd_args:
        print (key + ' ' + str(cmd_args[key]))

    # load data
    x = load_corpus(cmd_args['x_train'])
    y = load_labels(cmd_args['y_train'])
    # x_test = load_corpus(cmd_args['x_test'])
    # y_test = load_label(cmd_args['y_test'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # feature extraction
    x_train = extract_features(x_train, preprocess)
    x_test = extract_features(x_test, preprocess)


    feat = cmd_args['feat']
    if feat == "tf":
        vtr = CountVectorizer()
    elif feat == "tfidf":
        vtr = TfidfVectorizer()
    elif feat == "dict":
        vtr = DictVectorizer()

    model = cmd_args['model']
    if model == "nb":
        clf = MultinomialNB()
    elif model == "lr":
        clf = LogisticRegression()
    elif model == "svc":
        clf = SVC(probability=True)


    # build pipeline
    pl_steps = []
    pl_steps.append(('vtr',vtr))
    pl_steps.append(('clf',clf))
    pipeline = Pipeline(pl_steps)

    # fit and predict
    pipeline.fit(x_train, y_train)

    y_pred_prob = pipeline.predict_proba(x_test)
    y_pred_prob = [y[1] for y in y_pred_prob]
    y_pred = pipeline.predict(x_test)

    cm = metrics.confusion_matrix(y_test, y_pred)
    # micro = metrics.f1_score(y_test, y_pred, average='micro')
    # macro = metrics.f1_score(y_test, y_pred, average='macro')
    # weighted = metrics.f1_score(y_test, y_pred, average='weighted')

    binary = metrics.f1_score(y_test, y_pred)

    rmse = metrics.mean_squared_error(y_test, y_pred_prob)**0.5

    print ('confusion matrix:\n\n%s\n\n' % cm)
    # print ('micro f1: %s\n' % micro)
    # print ('macro f1: %s\n' % macro)
    # print ('weighted f1: %s\n' % weighted)
    if binary:
        print ('binary f1: %s\n' % binary)
    print ('rmse: %.4f\n' % rmse)


    # start of validation data results

    x_valid = load_corpus(cmd_args['x_test'])
    x_valid = extract_features(x_valid, preprocess)

    y_pred_prob = pipeline.predict_proba(x_valid)
    y_pred_prob = [y[1] for y in y_pred_prob]

    # write to output file
    write_submission(cmd_args['results'], y_pred_prob)


# headers = ['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3','short_description', 'price','product_type']

# input data
def extract_features(data, preprocess):
    features = []

    for row in data:

        country = row[0]
        title = row[2]
        category_lvl_1 = row[3]
        category_lvl_2 = row[4]
        category_lvl_3 = row[5]
        short_description = row[6]
        price = row[7]
        product_type = row[8]

        if preprocess:
            title = pp.process_data(title)

        soup = BeautifulSoup(title, "html.parser")
        text = soup.get_text()
        # if preprocess:
        #     short_description = pp.process_data(short_description)
        #
        # soup = BeautifulSoup(short_description, "html.parser")
        # text = soup.get_text()

        features.append(text)


    return features



# create submission text file in ../../submissions
def write_submission(filename, predicted_results):

    with open(filename, 'w') as f:
        for result in predicted_results:
            f.write(str(result) + '\n')
    f.close()


# takes in txt file of with each row representing a label
# returns list of label
def load_labels(label_file):

    y = []

    with open(label_file, 'r') as f:
        for line in f:
            label = line.strip()
            y.append(int(label))
    f.close()

    return y

# takes in gzipped file with rows of csv
# returns list of list of elements that were comma separated 2
def load_corpus(data_file):

    X = []

    with gzip.open(data_file, 'rb') as f:
        for line in f:
            line_arr = line.decode('utf8').split(',')
            X.append(line_arr)
    f.close()

    return X


if __name__ == '__main__':
    run()
