"""
An experiment of Classification Problem in Machine Learning
The program compares the precision of different methods
Including K-nn Algorithm with and without Weighted, Naive Bayesian Classification and SVM

2016-5-21
copyright by Kyle.Yang
"""


from numpy import *
from scipy import *
from svmutil import *
from matplotlib import pyplot 
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve  
import operator

#--------Input File Function--------#
def file2Mat(FileName, parammterNumber):
    fr = open(FileName)
    lines = fr.readlines()
    lineNums = len(lines)
    resultMat = zeros((lineNums, parammterNumber))
    for i in range(lineNums):
        line = lines[i].strip()
        itemMat = line.split(',')
        resultMat[i, :] = itemMat[0:parammterNumber]
    fr.close()
    return resultMat
    
def importWeighted(FileName):
    fr = open(FileName)
    WeightedList = list()
    for i in fr:
        i = abs(float(i))
        WeightedList.append(i)
    
    WeightedArray = array(WeightedList)
    ArgArray = argsort(WeightedArray)[::-1]
    return WeightedArray, ArgArray
#--------End of Input File Function--------#    

#--------------K-nn Algorithm---------------#
class KnnClassifier(object):
    
    def __init__(self, labels, samples):
        """ Initialize classifier with training data. """
        
        self.labels = labels
        self.samples = samples
    
    def classify(self, point, disAlgorithm, k):
        """ Classify a point against k nearest 
            in the training data, return label. """
        
        # compute distance to all training points
        if disAlgorithm == 1:
            dist = array([L1dist(point,s) for s in self.samples])
        if disAlgorithm == 2:
            dist = array([L2dist(point,s) for s in self.samples])
        
        # sort them
        ndx = dist.argsort()
        
        # use dictionary to store the k nearest
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label,0)
            votes[label] += 1
            
        return max(votes)

# No Weighted Distance #
def L1dist(p1,p2):
    return sqrt(sum((p1 - p2) ** 2))
# Weighted Distance #
def L2dist(p1,p2):
    return sqrt(sum(((p1 - p2) ** 2) * SubWeighted))
    
#--------------End of K-nn Algorithm----------------#    

#--------------Test of Accuracy---------------------#    
def testAccuracy(model, rows, disAlgorithm, param_k):
    correct = 0
    for i in range(rows / 2):
        if model.classify(testSubMat[i], disAlgorithm, param_k) == 'A':
            correct += 1
        if model.classify(testSubMat[i+400], disAlgorithm, param_k) == 'B':
            correct += 1
    
    print 'Accuracy = ' + str(correct / 800.0 * 100) + '%   |',
#---------------------------------------------------#      

if __name__ == '__main__':
    MTLWeighted, MTLLabel = importWeighted('data/label/MTL_Male.dat')
    CMTLWeighted, CMTLLabel = importWeighted('data/label/CMTL_Male.dat')
    CEMTLWeighted, CEMTLLabel = importWeighted('data/label/CEMTL_Male.dat')
    trainingMat = file2Mat('data/train/MTL_Male_train.dat', 3304)
    testMat = file2Mat('data/test/MTL_Male_test.dat',3304)

    print 'Select the Algorithm'
    print '1.K-nn Algorithm 2.K-nn Weighted Algorithm 3.SVM Algorithm 4.Naive Bayesian Classification 0.Exit'
    choice = input()
    while choice != 0:
        if choice == 1:
            print 'How many attributes mattered (0 for exit): ',
            NumberOfMatter = input()
            while NumberOfMatter != 0:
                print 'The parameter of K: ',
                param_k = input()
                print 'MTL                 |CMTL                |CEMTL                |'

                #--------MTL Label------#
                trainingSubMat = trainingMat[:,MTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,MTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                labels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                model = KnnClassifier(labels, trainingArray)
                testAccuracy(model, shape(testSubMat)[0], 1, param_k)
                #-----------------------#

                #-------CMTL Label------#
                trainingSubMat = trainingMat[:,CMTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,CMTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                labels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                model = KnnClassifier(labels, trainingArray)
                testAccuracy(model, shape(testSubMat)[0], 1, param_k)
                #-----------------------#

                #------CEMTL Label------#
                trainingSubMat = trainingMat[:,CEMTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,CEMTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                labels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                model = KnnClassifier(labels, trainingArray)
                testAccuracy(model, shape(testSubMat)[0], 1, param_k)
                #-----------------------#
                print
                print 'How many attributes mattered (0 for exit): ',
                NumberOfMatter = input()
            #----End of NumberOfMatter----#
        #----End of Choice 1----#
        
        if choice == 2:
            print 'How many attributes mattered (0 for exit): ',
            NumberOfMatter = input()
            while NumberOfMatter != 0:
                print 'The parameter of K: ',
                param_k = input()
                print 'MTL                 |CMTL                |CEMTL                |'

                #--------MTL Label------#
                trainingSubMat = trainingMat[:,MTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,MTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                SubWeighted = MTLWeighted[MTLLabel[0:NumberOfMatter]]
                labels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                model = KnnClassifier(labels, trainingArray)
                testAccuracy(model, shape(testSubMat)[0], 2, param_k)
                #-----------------------#

                #-------CMTL Label------#
                trainingSubMat = trainingMat[:,CMTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,CMTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                SubWeighted = MTLWeighted[MTLLabel[0:NumberOfMatter]]
                labels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                model = KnnClassifier(labels, trainingArray)
                testAccuracy(model, shape(testSubMat)[0], 2, param_k)
                #-----------------------#

                #------CEMTL Label------#
                trainingSubMat = trainingMat[:,CEMTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,CEMTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                SubWeighted = MTLWeighted[MTLLabel[0:NumberOfMatter]]
                labels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                model = KnnClassifier(labels, trainingArray)
                testAccuracy(model, shape(testSubMat)[0], 2, param_k)
                #-----------------------#
                print
                print 'How many attributes mattered (0 for exit): ',
                NumberOfMatter = input()
            #----End of NumberOfMatter----#
        #----End of Choice 2----#
        
        if choice == 3:
            param = svm_parameter()
            param.kernel_type = LINEAR
            print 'How many attributes mattered (0 for exit): ',
            NumberOfMatter = input()
            while NumberOfMatter != 0:
                print '1: MTL   2:CMTL  3:CEMTL 0:Exit  Label_Choice: ',
                LabelChoice = input()
                while LabelChoice != 0:
                    if LabelChoice == 1:
                        trainingSubMat = trainingMat[:,MTLLabel[0:NumberOfMatter]]
                        testSubMat = testMat[:,MTLLabel[0:NumberOfMatter]]
                    if LabelChoice == 2:
                        trainingSubMat = trainingMat[:,CMTLLabel[0:NumberOfMatter]]
                        testSubMat = testMat[:,CMTLLabel[0:NumberOfMatter]]
                    if LabelChoice == 3:
                        trainingSubMat = trainingMat[:,CEMTLLabel[0:NumberOfMatter]]
                        testSubMat = testMat[:,CEMTLLabel[0:NumberOfMatter]]
                    trainingLabels = [-1] * (shape(trainingSubMat)[0] / 2) + [1] * (shape(trainingSubMat)[0] / 2)
                    testLabels = [-1] * (shape(testSubMat)[0] / 2) + [1] * (shape(testSubMat)[0] / 2)
                    trainingSubList = list()
                    testSubList = list()
                    for i in range(shape(trainingSubMat)[0]):
                        trainingSubList.append(list(trainingMat[i]))
                    for i in range(shape(testSubMat)[0]):
                        testSubList.append(list(testMat[i]))
                    prob = svm_problem(trainingLabels, trainingSubList)
                    param = svm_parameter('-t 0 -c 4 -b 1')
                    m = svm_train(prob, param)
                    svm_predict(testLabels, testSubList, m)
                    print '1: MTL   2:CMTL  3:CEMTL 0:Exit  Label_Choice: ',
                    LabelChoice = input()
                #--------End of LabelChoice-------#
                print 'How many attributes mattered (0 for exit): ',
                NumberOfMatter = input()
            #----End of NumberOfMatter----#
        #----End of Choice 3----#
        
        if choice == 4:
            print 'How many attributes mattered (0 for exit): ',
            NumberOfMatter = input()
            while NumberOfMatter != 0:
                print 'MTL                 |CMTL                |CEMTL                |'

                #--------MTL Label------#
                trainingSubMat = trainingMat[:,MTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,MTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                testArray = array(testSubMat)
                traininglabels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                testlabels = ['A'] * (shape(testSubMat)[0] / 2) + ['B'] * (shape(testSubMat)[0] / 2)
                clf = MultinomialNB().fit(trainingArray, traininglabels)
                precision = mean(clf.predict(testArray) == testlabels) * 100
                print 'Accuracy = ' + str(precision) + '%   |', 
                #-----------------------#

                #-------CMTL Label------#
                trainingSubMat = trainingMat[:,CMTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,CMTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                testArray = array(testSubMat)
                traininglabels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                testlabels = ['A'] * (shape(testSubMat)[0] / 2) + ['B'] * (shape(testSubMat)[0] / 2)
                clf = MultinomialNB().fit(trainingArray, traininglabels)
                precision = mean(clf.predict(testArray) == testlabels) * 100
                print 'Accuracy = ' + str(precision) + '%   |',
                #-----------------------#

                #------CEMTL Label------#
                trainingSubMat = trainingMat[:,CEMTLLabel[0:NumberOfMatter]]
                testSubMat = testMat[:,CEMTLLabel[0:NumberOfMatter]]
                trainingArray = array(trainingSubMat)
                testArray = array(testSubMat)
                traininglabels = ['A'] * (shape(trainingSubMat)[0] / 2) + ['B'] * (shape(trainingSubMat)[0] / 2)
                testlabels = ['A'] * (shape(testSubMat)[0] / 2) + ['B'] * (shape(testSubMat)[0] / 2)
                clf = MultinomialNB().fit(trainingArray, traininglabels)
                precision = mean(clf.predict(testArray) == testlabels) * 100
                print 'Accuracy = ' + str(precision) + '%   |',
                #-----------------------#
                print
                print 'How many attributes mattered (0 for exit): ',
                NumberOfMatter = input()
            #----End of NumberOfMatter----#
        #----End of Choice 4----#
        
        print 'Select the Algorithm'
        print '1.K-nn Algorithm 2.K-nn Weighted Algorithm 3.SVM Algorithm 4.Naive Bayesian Classification 0.Exit'
        choice = input()
    #----End of Choice----#
