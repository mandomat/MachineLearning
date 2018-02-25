import random
class dummy_classifier():
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predictions =[]
        for row in x_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions
