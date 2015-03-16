# NaiveBayesSpamClassifier
This is a practice of spam detector based on Naive Bayes Classifier
written by Matlab. The framework is provided by Prof. L. Hellerstein.
This program has to be run on the dataset enron.mat.

The datafile contains three parts- training, validation, and test. Each
part contains two sets, one with information about the words appearing
in the emails in that set, and one with information about the labels of
those emails.The data set with the term (feature) information contains
a DxW matrix, in sparse format, where D is the number of
emails in the set. Each row of the matrix corresponds to an email in
the set, each column corresponds to a vocabulary term, and entry [i,j]
of the matrix contains the number of occurrences of vocabulary term j
in email number i in the set. The
file with the label information contains a Dx1 matrix where the ith
entry is 1 if email number i is spam, and 0 if it is ham.
