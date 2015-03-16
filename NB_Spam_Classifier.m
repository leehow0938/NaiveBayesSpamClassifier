%% Spam detection exercise - ENRON EMAILS
%% Created by L. Hellerstein and Edison Zhao
%% Modified by Hao Li (N12238066)
%% Date: 3/10/2015
%% The version of the Enron dataset used in the data file enron.mat
%% was taken from Homework 1 of CSCI1420, Brown University, 2013
%% cs.brown.edu/courses/csci1420/

clear all; close all; clc;
format long;

% Load the data from the course directory
load('enron.mat');

% VARIABLES IN 'enron.mat'
% trainFeat: sparse matrix of word counts for training documents.
% trainLabels: matrix of {0,1} training labels where 0=ham,1=spam.
% testFeat: sparse matrix of word counts for test documents.
% testLabels: matrix of test document labels.
% vocab: cell array giving word (character string) for each vocabulary index.

% valFeat: sparse matrix of word counts for validation documents. (we don't use)
% valLabels: matrix of validation document labels. (we don't use)

%% "Binarize" the matrices trainFeat and testFeat by replacing all positive entries with the number 1

trainFeat = spones(trainFeat);
testFeat = spones(testFeat);

%% Calculate the total number of spam and ham emails
total_train = length(trainLabels);
num_spam = sum(trainLabels);
num_ham = total_train-num_spam;

%% Calculate the prior proabilities of ham and spam.  No smoothing.
spam_prior = num_spam/total_train;
ham_prior = num_ham/total_train;

%% Calculate total number of spam emails containing each term w, then do the same for ham
spam_w_freq = trainLabels' * trainFeat; 
ham_w_freq = ~trainLabels' * trainFeat;

%% For each term w, compute the estimate of P(w|spam), 
%% the probability that w occurs, given that the email is spam.
%% Also, for each term w, compute the estimate of P(w|ham).
%% Similarly, compute the estimate of P(not w|spam)
%% and P(not w|ham), the conditional probabilities that w does not occur.
%% Remember to smooth!  Since the feature corresponding
%% to w has 2 possible values, true or false, t=2 and k=0.1 in our smoothing formula.
k=0.1;
t=2;
spam_w_prob = (spam_w_freq + k)/(num_spam + k * t);
ham_w_prob =  (ham_w_freq + k)/(num_ham + k * t);

spam_notw_prob = (num_spam - spam_w_freq + k)/(num_spam + k * t);
ham_notw_prob = (num_ham - ham_w_freq + k)/(num_ham + k * t);

%% Calculate the logs of P(w|spam) and P(w|ham)
spam_w_log_prob = log(spam_w_prob);
ham_w_log_prob = log(ham_w_prob);

%% Calculate the logs of P(not w|spam) and P(not w|ham)
spam_notw_log_prob = log(spam_notw_prob);
ham_notw_log_prob = log(ham_notw_prob);

% Using the values computed above, for all emails Y in the training set,
% we want to calculate log P(spam|Y)*P(spam) and log P(ham|Y)*P(ham).
%
% To do this, we will use that fact that if
% x is a variable that is 1 when w occurs in a given email, and 0 when w
% does not occur, then 
% log P(x|spam) = x*log P(w|spam) + (1-x)*log P(not w|spam).
% Multiplying out the second term, and rearranging, we get
%
% P(x|spam) = x*log P(w|spam) + (log P(not w|spam) - x*log P(not w|spam)) 
%
% By multiplying out this way, we avoid the computation of (1-x).  This is
% important because we will be subsituting trainFeat for x, and we do not
% want to calculate (1-trainFeat), the bitwise complement of the matrix
% trainFeat.  It has too many non-zero entries and calculating it would cause our
% runtime to be slow.  Also note that the second term in the above
% expression is independent of x.

sum_spam_notw_log_prob = sum(spam_notw_log_prob);

% For each training email,
% calculate the sum of (log P(not w|spam) - x*log P(not w|spam)) over all words w, where x=1 if w
% occurs, and x=0 otherwise.
%
% In the next line, note that sum_spam_notw_log_prob is a scalar that corresponds to
% log P(not w|spam), which is independent of x.  It is added to all entries
% of the matrix -(spam_notw_log_prob*transpose(trainFeat))

spam_notw_train_term = sum_spam_notw_log_prob - spam_notw_log_prob * transpose(trainFeat);

% Calculate log P(email|spam) + log(spam), for each training email.
% log(spam_prior) is a scalar that is added to all entries of the computed
% vector
spam_prob_train = (spam_w_log_prob * transpose(trainFeat) + spam_notw_train_term) + log(spam_prior);

% Do the analogous computation for ham_prob_train
sum_ham_notw_log_prob = sum(ham_notw_log_prob);
ham_notw_train_term = sum_ham_notw_log_prob - ham_notw_log_prob * transpose(trainFeat);
ham_prob_train = (ham_w_log_prob * transpose(trainFeat) + ham_notw_train_term) + log(ham_prior);

predict_result_train = (spam_prob_train > ham_prob_train)';

%% The accuracy obtained on the TRAINING set (accuracy = 1 - error) 
eer_train = nnz(predict_result_train - trainLabels);

disp('Accuracy on Training Set:');
1 - eer_train/length(trainLabels)

%% Now compute accuracy on the TEST set, using same approach
spam_notw_test_term = sum_spam_notw_log_prob - spam_notw_log_prob * transpose(testFeat);
ham_notw_test_term = sum_ham_notw_log_prob - ham_notw_log_prob * transpose(testFeat);

spam_prob_test = (spam_w_log_prob * transpose(testFeat) + spam_notw_test_term) + log(spam_prior);
ham_prob_test = (ham_w_log_prob * transpose(testFeat) + ham_notw_test_term) + log(ham_prior);

predict_result_test = (spam_prob_test > ham_prob_test)';

eer_test = nnz(predict_result_test - testLabels);
%% The accuracy obtained on the TEST set
disp('Accuracy on Test Set:')
1 - eer_test/length(testLabels)
