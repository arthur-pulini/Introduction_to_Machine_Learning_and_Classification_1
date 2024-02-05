from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()

#features (1 yes, 0 no)
# long hair
# short leg
# bark
pig1 = [0, 1, 0]
pig2 = [0, 1, 1]
pig3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

# 1 => pig , 0 => dog
trainX = [pig1, pig2, pig3, dog1, dog2, dog3]
trainY = [1,1,1,0,0,0]

model.fit(trainX, trainY)

mysterious1 = [1,1,1]
mysterious2 = [1,1,0]
mysterious3 = [0,1,1]

testsX = [mysterious1, mysterious2, mysterious3]
testsY = [0,1,1]

predictions = model.predict(testsX)
print(predictions)

accuracyScore = accuracy_score(testsY, predictions)
print("The accuracy was: %.2f " % (accuracyScore * 100))
