import numpy as np
import pandas as pd
from csv import reader as rdr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pdb


class MyLogistechRegression:
  def __init__ (self,p_learningRate,p_iterations):
    self.m_learningRate = p_learningRate
    self.m_iterations = p_iterations

  def shuffle(self, a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a, b
  
  def sigmoid(self,tmpX):
    return 1.0 / (1 + np.exp(-tmpX))

  def declareVariable(self,x,y):
    self.inputSample = x.shape[0]
    self.inputFeatures = x.shape[1]
    self.weight_matrix = np.zeros(self.inputFeatures)
    self.bias = 0

  def calculateValueTraining(self,x, y, reshapeFactor):
    calculated_value = np.dot(x,self.weight_matrix) + self.bias
    calculated_value_with_sigmoid = self.sigmoid(calculated_value)
    tmpDiff = calculated_value_with_sigmoid - y.T
    tmpDiff = np.reshape(tmpDiff,reshapeFactor)
    return calculated_value_with_sigmoid, tmpDiff

  def calculateDecreaseRateTraining(self,x,inputSamples,difference):
    return (1/inputSamples)*np.dot(x.T,difference), (1/inputSamples)*sum(difference)
  
  def updateWeightAndBiasTraining(self,dw,db):
    self.weight_matrix = self.weight_matrix - (self.m_learningRate*dw)
    self.bias = self.bias - (self.m_learningRate*db)

  def calculateValueValidation(self,x_validation, y_validations):
    calculated_value_validation = np.dot(x_validation,self.weight_matrix) + self.bias
    calculated_value_with_sigmoid_validation = self.sigmoid(calculated_value_validation)
    tmpDiffValidation = calculated_value_with_sigmoid_validation - y_validations.T
    return calculated_value_with_sigmoid_validation, tmpDiffValidation

  def setGraphVariablesbgd(self,differenceTraining,differenceValidation,validationSize):
    self.y_training_loss_bgd.append(np.sqrt(np.sum(np.square(differenceTraining)))/self.inputSample)
    self.y_validation_loss_bgd.append(np.sqrt(np.sum(np.square(differenceTraining)))/validationSize)

  def setGraphVariablesSgd(self,differenceTraining,differenceValidation,validationSize):
    self.y_training_loss_sgd.append(np.sqrt(np.sum(np.square(differenceTraining)))/self.inputSample)
    self.y_validation_loss_sgd.append(np.sqrt(np.sum(np.square(differenceTraining)))/validationSize)

  def bgdFitting(self, x, y, x_validation, y_validation):
    self.declareVariable(x,y)
    self.y_validation_loss_bgd = []
    self.y_training_loss_bgd = []

    i = 0
    while(i< self.m_iterations):
      calculated_value_with_sigmoid_training, tmpDiffTraining = self.calculateValueTraining(x,y,self.inputSample)
      calculated_value_with_sigmoid_validation, tmpDiffValidation = self.calculateValueValidation(x_validation,y_validation)

      dw,db = self.calculateDecreaseRateTraining(x,self.inputSample,tmpDiffTraining)

      self.setGraphVariablesbgd(tmpDiffTraining,tmpDiffValidation,x_validation.shape[0])

      self.updateWeightAndBiasTraining(dw,db)
      i += 1
    return self.y_training_loss_bgd, self.y_validation_loss_bgd
  
  def sgdFitting(self, x, y, x_validation, y_validation, sgdVal):
    self.declareVariable(x,y)
    self.y_validation_loss_sgd = []
    self.y_training_loss_sgd = []

    i = 0
    while(i < self.m_iterations):
      tmpX, tmpY = self.shuffle(x,y)
      tmpX = tmpX[:sgdVal]
      tmpY = tmpY[:sgdVal]

      calculated_value_with_sigmoid_training, tmpDiffTraining = self.calculateValueTraining(tmpX,tmpY,sgdVal)
      calculated_value_with_sigmoid_validation, tmpDiffValidation = self.calculateValueValidation(x_validation,y_validation)

      dw,db = self.calculateDecreaseRateTraining(tmpX,self.inputSample,tmpDiffTraining)

      self.setGraphVariablesSgd(tmpDiffTraining,tmpDiffValidation,x_validation.shape[0])

      self.updateWeightAndBiasTraining(dw,db)
      i += 1
    return self.y_training_loss_sgd, self.y_validation_loss_sgd

  def predict(self,x):
    calculate_val = np.dot(x,self.weight_matrix) + self.bias
    return np.where(calculate_val > 0.5, 1, 0)

  def iterationVsLossGraph(self, type):
    x_axis = [i for i in range(0, self.m_iterations)]
    if(type == 'bgd'):
      plt.plot(x_axis, self.y_training_loss_bgd, 'blue', label = "Training")
      plt.plot(x_axis, self.y_validation_loss_bgd, 'red', label = "Validation")
      plt.title('BGD Loss Plots')
    if(type == 'sgd'):
      plt.plot(x_axis, self.y_training_loss_sgd, 'blue', label = "Training")
      plt.plot(x_axis, self.y_validation_loss_sgd, 'red', label = "Validation")
      plt.title('SGD Loss Plots')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()



class Evaluation:
  def iterationVsLossGraph(self, type, iterations, y_training_loss, y_validation_loss):
    x_vals = [i for i in range(0, iterations)]
    plt.plot(x_vals, y_training_loss, 'blue', label = "Training")
    plt.plot(x_vals, y_validation_loss, 'red', label = "Validation")
    if(type == 'bgd'):
      plt.title('BGD Loss Plot with number of iterations = '+str(iterations))
    if(type == 'sgd'):
      plt.title('SGD Loss Plot with number of iterations = '+str(iterations))
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()
  
  def confusionMatrix(self, actual_output, predicted_output):
    true_positive = 0
    true_negatives = 0
    false_positives = 0
    false_negative = 0

    assert len(actual_output) == len(predicted_output)
    i = 0
    while(i < len(actual_output)):
      if((actual_output[i] == 1) and (predicted_output[i] == 1)):
        true_positive += 1
      if((actual_output[i] == 1) and (predicted_output[i] == 0)):
        true_negatives += 1
      if((actual_output[i] == 0) and (predicted_output[i] == 1)):
        false_positives += 1
      if((actual_output[i] == 0) and (predicted_output[i] == 0)):
        false_negative += 1
      i += 1
    
    return true_positive,true_negatives,false_positives,false_negative

  def accuracyScore(self, true_positive, true_negatives, false_positives, false_negative):
    return (true_positive + true_negatives) / (true_positive + true_negatives + false_positives + false_negative)

  def recallScore(self, true_positive, true_negatives, false_positives, false_negative):
    result = 0
    try:
      result = (true_positive) / (true_positive + false_negative)
      return result
    except ZeroDivisionError:
      print("ZeroDivisionError")

  def precisionScore(self, true_positive, true_negatives, false_positives, false_negative):
    result = 0
    try:
      result = (true_positive) / (true_positive + false_positives)
      return result
    except ZeroDivisionError:
      print("ZeroDivisionError")

  def F1Score(self, true_positive, true_negatives, false_positives, false_negative):
    result = 0
    try:
      result = (2 * true_positive) / ((2 * true_positive) + false_positives + false_negative)
      return result
    except ZeroDivisionError:
      print("ZeroDivisionError")

def readDataset(filename):
  df = pd.read_csv(filename)
  y = list(df['Outcome'])
  x_labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
  df = df[x_labels]
  x = []
  for ind,row in df.iterrows():
    x.append(list(row))
  return x, y

def divideDataset(dataset):
  split = np.split(dataset,[int(0.7*len(dataset)), int(0.9*len(dataset))])
  return split[0],split[1],split[2]

def computeResults(datasetInput):
  x,y  = readDataset(datasetInput)
  x_train,x_validate,x_test = divideDataset(x)
  y_train,y_validate,y_test = divideDataset(y)

  myEvaluation = Evaluation()

  learning_rate = [0.01, 0.0001, 10]

  for val in learning_rate:
    print("Learning Rate:  "+str(val))

    model = MyLogistechRegression(val, 1000)

    y_training_loss_bgd, y_validation_loss_bgd = model.bgdFitting(x_train,y_train,x_validate,y_validate)
    bgd_predicted_values = list(model.predict(x_test))
    
    y_training_loss_sgd, y_validation_loss_sgd = model.sgdFitting(x_train,y_train,x_validate,y_validate,200)
    sgd_predicted_values = list(model.predict(x_test))

    true_positive_bgd, true_negatives_bgd, false_positives_bgd, false_negative_bgd = myEvaluation.confusionMatrix(list(y_test), bgd_predicted_values)
    accuracy_bgd = myEvaluation.accuracyScore(true_positive_bgd, true_negatives_bgd, false_positives_bgd, false_negative_bgd)
    precision_bgd = myEvaluation.precisionScore(true_positive_bgd, true_negatives_bgd, false_positives_bgd, false_negative_bgd)
    recall_bgd = myEvaluation.recallScore(true_positive_bgd, true_negatives_bgd, false_positives_bgd, false_negative_bgd)
    f1_bgd = myEvaluation.F1Score(true_positive_bgd, true_negatives_bgd, false_positives_bgd, false_negative_bgd)

    true_positive_sgd, true_negatives_sgd, false_positives_sgd, false_negative_sgd = myEvaluation.confusionMatrix(list(y_test), sgd_predicted_values)
    accuracy_sgd = myEvaluation.accuracyScore(true_positive_sgd, true_negatives_sgd, false_positives_sgd, false_negative_sgd)
    precision_sgd = myEvaluation.precisionScore(true_positive_sgd, true_negatives_sgd, false_positives_sgd, false_negative_sgd)
    recall_sgd = myEvaluation.recallScore(true_positive_sgd, true_negatives_sgd, false_positives_sgd, false_negative_sgd)
    f1_sgd = myEvaluation.F1Score(true_positive_sgd, true_negatives_sgd, false_positives_sgd, false_negative_sgd)

    print()
    print()
    print("confusion matrix BGD")
    print(str(true_positive_bgd) +"       "+str(false_positives_bgd))
    print(str(false_positives_bgd)+"       "+str(false_negative_bgd))
    print()
    print("ACCURACY SCORE BGD: "+str(accuracy_bgd))
    print("PRECISION SCORE BGD:  "+str(precision_bgd))
    print("RECALL SCORE BGD:   "+str(recall_bgd))
    print("F1 SCORE BGD:   "+str(f1_bgd))
    print()
    print()
    print("confusion matrix SGD")
    print(str(true_positive_sgd) +"       "+str(false_positives_sgd))
    print(str(false_positives_sgd)+"       "+str(false_negative_sgd))
    print()
    print("ACCURACY SCORE SGD: "+str(accuracy_sgd))
    print("PRECISION SCORE SGD:  "+str(precision_sgd))
    print("RECALL SCORE SGD:   "+str(recall_sgd))
    print("F1 SCORE SGD:   "+str(f1_sgd))
    myEvaluation.iterationVsLossGraph('BGD',1000,y_training_loss_bgd,y_validation_loss_bgd)
    myEvaluation.iterationVsLossGraph('SGD',1000,y_training_loss_sgd,y_validation_loss_sgd)
    print()
    print()
    print()
    print()

def compareResults(datasetInput):
  x,y  = readDataset(datasetInput)
  x_train,x_validate,x_test = divideDataset(x)
  y_train,y_validate,y_test = divideDataset(y)

  sklearnLogisticRegression = LogisticRegression()
  sklearnLogisticRegression.fit(x_train,y_train)
  sklearPredict = sklearnLogisticRegression.predict(x_test)
  
computeResults("C:\\Users\\Keshav Gambhir\\Desktop\\Assignment-1\\Datasets\\diabetes2.csv")