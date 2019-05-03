# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    print(legalLabels)
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
      #print(self.weights[label])
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    lab = util.Counter()

    for iteration in range(self.max_iterations):
      print ("Starting iteration ", iteration, "...")
      for i in range(len(trainingData)):
        for l in self.legalLabels:
          #print(trainingData[i])
          lab[l] = trainingData[i].__mul__(self.weights[l])
        if not(trainingLabels[i] == lab.argMax()):
          self.weights[trainingLabels[i]].__radd__(trainingData[i])
          self.weights[lab.argMax()].__sub__(trainingData[i])
          "*** YOUR CODE HERE ***"
          #util.raiseNotDefined()
    
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
    Returns a list of the 100 features with the greatest difference in weights:
                     w_label1 - w_label2

    """
    featuresOdds = []
    featuresWeights = self.weights[label].values()
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds

