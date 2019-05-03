import os

def callBasedOnClassifier(classifier):
    if classifier == "knear":
        classifier = classifier+" -f"
        doCalls(classifier)
    else:
        doCalls(classifier)

def doCalls(classifier):
    x = 10
    while x <= 100:
        command = "python2 dataClassifier.py -c " + classifier + " -t 1000 -s 1000 -p " + str(x)
        os.system(command)
        x = x + 10
    '''y = 10
    while y <= 100:
        command = "python dataClassifier.py -c " + classifier + " -d faces -t 450 -s 150 -p " + str(y)
        os.system(command)
        y = y + 10'''


classifiers = ["perceptron"]

#overwrite existing log.txt
f = open('log.txt','w')
f.close()
#for c in classifiers:
f = open('log.txt','a')
f.write(classifiers[0] + " ")
f.close()
callBasedOnClassifier(classifiers[0])