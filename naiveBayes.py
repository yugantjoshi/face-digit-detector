# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    bestcount = -1
    conditional_all = util.Counter()
    prior_all = util.Counter()#this is used to keep track of
    feature_total = util.Counter()

    #initializing conditional_all to 1 for all fields
    for label in self.legalLabels:
        for feat in self.features:
            # CHANGE 2 when values for features change
            feature_total[feat,label] = 2
            for value in range(0,2):
                conditional_all[feat, label, value] = 1

    #filling in data to the structures
    for i in range(len(trainingData)):
        img = trainingData[i]
        label = trainingLabels[i]
        prior_all[label] += 1
        for feat, value in img.items():
            feature_total[(feat, label)] += 1
            conditional_all[(feat, label, value)] += 1
                #if label == 0:
                 #   print "wth", feat, conditional_all[feat, label, value]

    for k in kgrid:
        # copying items from main variables to temp variables
        # because we are gonna change them for each value of k
        conditional_temp = util.Counter()
        prior_temp = util.Counter()
        features_temp = util.Counter()

        for key, val in conditional_all.items():
          conditional_temp[key] += val
        for key, val in prior_all.items():
          prior_temp[key] += val
        for key, val in feature_total.items():
          features_temp[key] += val


        #smoothing by adding k
        # statement below basically does c(f, y) + k for all all values that a feature takes
        for key, val in conditional_all.items():
            conditional_temp[key] = val + k
        #statement below basically adds k to all values corresponding to that feature and label (thats why 2 for binary)
        for key, val in feature_total.items():
            features_temp[key] = val + k*2  # increase this value if features take more than two values (2 rn because binary)



        # now calculate c(fi,y)/c(F,y)
        for key, val in conditional_temp.items():
            features_key = (key[0], key[1])
            conditional_temp[key] = val*1.0/features_temp[features_key]

        self.conditionalProb = conditional_temp

        #print conditional_temp

        # normalize the prior
        prior_temp.normalize()
        self.prior = prior_temp

        self.k = k
        # evaluating performance on validation set
        predictions = self.classify(validationData)
        accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        print "Performance on validation set for k=%f: (%.1f%%)" % (k, 100.0 * accuracyCount / len(validationLabels))

        if accuracyCount > bestcount:
            bestcount = accuracyCount
            best_prior = prior_temp
            best_conditionalProb = conditional_temp
            best_k = k
        #print conditional_temp
    self.prior = best_prior
    self.conditionalProb = best_conditionalProb
    self.k = best_k
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    for label in self.legalLabels:
        logJoint[label] = math.log(self.prior[label])
        for feat, value in datum.items():
            # print label, feat, self.conditionalProb[feat,label,value]
            prob = self.conditionalProb[feat, label, value]
            logJoint[label] += math.log(prob)

    #print logJoint
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds