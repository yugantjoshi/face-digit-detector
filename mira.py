# Mira implementation
import util
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """
    greatest_weights = util.Counter()
    greatest_acc = float('-inf')

    for c in Cgrid:
      # Get current weights
      current_weight = self.weights.copy()
      # Suggested to go through max_iterations constant
      for iteration in range(self.max_iterations):
        for k,w in enumerate(trainingData):
          prediction_score = float('-inf')
          prediction_label = float('-inf')

          # Prediction with weight w
          for label in self.legalLabels:
            if w*current_weight[label] > prediction_score:
              prediction_score = w*current_weight[label]
              prediction_label = label

          # Check if this matches the true value
          real_label = trainingLabels[k]
          if real_label != prediction_label:
            # Update weight again
            f_val = w.copy()
            minimized_tau = min(c, ((current_weight[prediction_label]-current_weight[real_label])*f_val+1.0)/(2.0*(f_val*f_val)))
            f_val.divideAll(1.0/minimized_tau)
            current_weight[prediction_label] = current_weight[prediction_label]-f_val
            current_weight[real_label] = current_weight[real_label]+f_val
            # check the accuracy of given c
            guesses = self.classify(validationData)
            correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            accuracy = float(float(correct) / len(guesses))

            # update the weights, if accuracy is higher
            if accuracy > greatest_acc:
              greatest_acc = accuracy
              greatest_weights = current_weight

          # set the best weight values
          self.weights = greatest_weights

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses


  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    "*** YOUR CODE HERE ***"

    return featuresOdds

