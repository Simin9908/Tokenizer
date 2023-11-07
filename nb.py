import math
from collections import Counter
import sys


def tokenize(text, n=3):
    if method == 'nbse':
        text = text.lower()
        text = "<"+text+">"
        return [text[i:i+n] for i in range(len(text) - (n-1))]
    else:
        text = text.lower()
        return [text[i:i+n] for i in range(len(text) - (n-1))]


def NB(data):
    # calculate probability of name for each country and store in dict
    prob_dict = {}
    document = tokenize(data)  # names tokenized into trigrams
    country_class = ""  # predicted country
    # calculate P(c|d) for each country class
    inc = 0
    ND = len(class_list)  # names in the document
    while inc < len(countries):
        country_class = countries[inc]
        # next we need prior for this class and P(f1,f2,f3|c)
        NC = nc_dict.get(country_class)
        prior = NC/ND
        prior_log = math.log(prior)
        # now we calc P(d|c) or likelihood
        likelihood = 0  # default value
        # check if each word is there in the training set/vocab
        for d in document:
            if d in vocab_dict:
                # count how many times its there in russian class
                d_count = nested_list[inc].count(d)
                temp_lh = (d_count+1) / \
                    (len(nested_list[inc]) + len(vocab_dict))
                temp_lh = math.log(temp_lh)
                likelihood = likelihood + temp_lh  # sum of log values
        probability = prior_log + likelihood
        prob_dict[countries[inc]] = probability
        inc = inc + 1
    country_class = max(prob_dict, key=prob_dict.get)
    return country_class


def NBSE(data):
    prob_dict = {}
    document = tokenize(data)
    country_class = ""
    inc = 0
    ND = len(class_list)
    while inc < len(countries):
        country_class = countries[inc]
        NC = nc_dict.get(country_class)
        prior = NC/ND
        prior_log = math.log(prior)
        likelihood = 0
        for d in document:
            if d in vocab_dict:
                # count how many times its there in russian class
                d_count = nested_list[inc].count(d)
                temp_lh = (d_count+1) / \
                    (len(nested_list[inc]) + len(vocab_dict))
                temp_lh = math.log(temp_lh)
                likelihood = likelihood + temp_lh  # sum of log values
        probability = prior_log + likelihood
        prob_dict[countries[inc]] = probability
        inc = inc + 1
    country_class = max(prob_dict, key=prob_dict.get)
    return country_class


class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True,
                          key=lambda x: klass_freqs[x])[0]

    def classify(self, test_instance):
        return self.mfc


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')

    method = sys.argv[1]
    nameFile = sys.argv[2]
    classFile = sys.argv[3]
    testFile = sys.argv[4]

    train_texts = [x.strip() for x in open(nameFile,
                                           encoding='utf8')]  # all the names
    class_list = [x.strip() for x in open(classFile,
                                          encoding='utf8')]  # all the countries
    test_data = [x.strip() for x in open(testFile,
                                         encoding='utf8')]  # test data will be stored in this list

    countries = []  # list of unique countries
    if (method == 'nb' or method == 'nbse'):
        count = 0
        while count < len(class_list):
            country = class_list[count]
            if country not in countries:
                countries.append(country)
            count = count + 1

        # list of list within list where 1 list consists of names from a particular country
        nested_list = [[] for _ in range(len(countries))]

        # for each name that is from a particular country - append it into the list
        country_inc = 0
        while country_inc < len(countries):
            country = countries[country_inc]
            i = 0
            while i < len(class_list):
                if class_list[i] == country:
                    nested_list[country_inc].append(train_texts[i])
                i = i+1
            country_inc = country_inc+1

        nc_dict = {}
        for c in range(len(countries)):
            nc_dict[countries[c]] = len(nested_list[c])
        # code gets split now between nb & nbse
        if method == 'nb':
            i = 0
            while i < len(nested_list):
                # nestlist[i] is all of the names belonging to one country
                tokenized_list = []
                for j in nested_list[i]:
                    # j is each russian name
                    tokened = tokenize(j)
                    for r in tokened:
                        # tokenized_list is list of 3letter russian names
                        tokenized_list.append(r)
                nested_list[i] = tokenized_list
                i = i+1
            vocab = [val for sublist in nested_list for val in sublist]
            vocab_dict = dict(Counter(vocab))
            for name in test_data:
                test_class = NB(name)  # CALL NB DEF BLOCK
                print(test_class)

        elif method == 'nbse':
            i = 0
            while i < len(nested_list):
                # nestlist[i] is all of the names belonging to one country
                tokenized_list = []
                for j in nested_list[i]:
                    # j is each name from selected country
                    tokened = tokenize(j)
                    for r in tokened:
                        tokenized_list.append(r)
                nested_list[i] = tokenized_list
                i = i+1
            vocab = [val for sublist in nested_list for val in sublist]
            # remove duplicates from the list
            vocab_dict = dict(Counter(vocab))
            for name in test_data:
                test_class = NBSE(name)  # CALL NBSE DEF BLOCK
                print(test_class)

    elif method == 'baseline':
        classifier = Baseline(class_list)
        results = [classifier.classify(x) for x in test_data]
        for r in results:
            print(r)

    elif method == 'lr':
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        count_vectorizer = CountVectorizer(analyzer=tokenize)
        train_counts = count_vectorizer.fit_transform(train_texts)
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, class_list)
        test_counts = count_vectorizer.transform(test_data)
        results = clf.predict(test_counts)
        for r in results:
            print(r)
