import pandas
import time
from algorithm import trainTestSplit, buildTree, predict, calculateAccuracy

dataFrame = pandas.read_csv("data/ai_data.csv")
dataFrame = dataFrame.drop("id", axis = 1)
dataFrame = dataFrame.drop("annual_inc", axis = 1)
dataFrame = dataFrame.drop("int_rate", axis = 1)
dataFrame = dataFrame.drop("loan_amnt", axis = 1)

dataFrame = dataFrame[dataFrame.columns.tolist()[1: ] + dataFrame.columns.tolist()[0: 1]]

termMapping = {"36 months": 1, "60 months": 2}
dataFrame["term"] = dataFrame["term"].map(termMapping)

gradeMapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
dataFrame["grade"] = dataFrame["grade"].map(gradeMapping)

homeOwnershipMapping = {"RENT": 1, "OWN": 2, "MORTGAGE": 3}
dataFrame["home_ownership"] = dataFrame["home_ownership"].map(homeOwnershipMapping)

verificationStatusMapping = {"Not Verified": 1, "Source Verified": 2, "Verified": 3}
dataFrame["verification_status"] = dataFrame["verification_status"].map(verificationStatusMapping)

loanStatusMapping = {"Charged Off": 1, "Current": 2, "Fully Paid": 3}
dataFrame["loan_status"] = dataFrame["loan_status"].map(loanStatusMapping)

purposeMapping = {"credit_card": 1, "car": 2, "small_business": 3, "wedding": 4, "major_purchase": 5, "home_improvement": 6, "debt_consolidation": 7, "other" : 8}
dataFrame["purpose"] = dataFrame["purpose"].map(purposeMapping)

dataFrameTrain, dataFrameTest = trainTestSplit(dataFrame, testSize = 0.3)

print("Decision Tree - Car Evaluation Dataset")

i = 1
accuracyTrain = 0
while accuracyTrain < 80:
    startTime = time.time()
    decisionTree = buildTree(dataFrameTrain, maxDepth = i)
    buildingTime = time.time() - startTime
    decisionTreeTestResults = predict(dataFrameTest, decisionTree)
    accuracyTest = calculateAccuracy(decisionTreeTestResults, dataFrameTest.iloc[:, -1]) * 100
    decisionTreeTrainResults = predict(dataFrameTrain, decisionTree)
    accuracyTrain = calculateAccuracy(decisionTreeTrainResults, dataFrameTrain.iloc[:, -1]) * 100
    print("maxDepth = {}: ".format(i), end = "")
    print("accTest = {0:.2f}%, ".format(accuracyTest), end = "")
    print("accTrain = {0:.2f}%, ".format(accuracyTrain), end = "")
    print("buildTime = {0:.2f}s".format(buildingTime), end = "\n")
    i += 1
