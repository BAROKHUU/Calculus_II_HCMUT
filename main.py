import math
import random
import numpy as np
from keras.datasets import mnist

def load_datasets():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    #Only 0 and 1 for classification
    train_mask = (y_train == 0) | (y_train == 1)
    test_mask = (y_test ==0) | (y_test == 1)

    x_train = x_train[train_mask]
    y_train = y_train[train_mask]

    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    #normalize from [0,255] to [0,1] <=> RGB to grayscale
    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0

    #image 28x28 pixels into 784 dims vector
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1) 

    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    return x_train, y_train, x_test, y_test

def set_randomseed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

#return predict probs of y_hat 
def predict_probability(X ,weights, bias):
    z = X @ weights + bias # @ matrix multiplication
    return sigmoid(z)


#predict label 0 or 1 from probs
def predict(X, weights, bias, threshold=0.5):
    probs = predict_probability(X, weights, bias)
    return (probs >= threshold).astype(np.int32) 
    #compare each element in probs w threshold then return true/false
    #then use np.int32 to convert into 0(false)/1(true)


#binary cross entropy
def loss_func(y_true, y_hat):
    epsilon = 1e-10

    y_hat = np.clip(y_hat, epsilon, 1-epsilon)
    #epsilon & 1-epsilon are limits of y_hat

    loss = -np.mean(y_true*np.log(y_hat) + (1.0 - y_true)*np.log(1.0-y_hat))

    return loss


def gradients_func(X, y_true, y_hat):
    
    m = X.shape[0]
    error = y_hat-y_true

    d_weights = (X.T @ error) / m
    d_bias = np.sum(error) / m

    return d_weights, d_bias


def accuracy_func(y_true, y_hat):
    return np.mean(y_true == y_hat)




#======================
#3. GRADIENT DESCENT
#======================

def Logistic_Regression_Training(X, y, learning_rate=0.12, epochs = 200):
    n_features = X.shape[1]

    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0

    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        y_hat = predict_probability(X, weights, bias)

        loss = loss_func(y, y_hat)

        d_weigths, d_bias = gradients_func(X, y, y_hat)

        weights = weights - learning_rate*d_weigths
        bias = bias - learning_rate*d_bias


        y_hat = predict(X, weights, bias)
        accuracy = accuracy_func(y, y_hat)


        loss_history.append(loss)
        accuracy_history.append(accuracy)

        if epoch == 0 or (epoch+1)%10==0:
            print(f"Epoch {epoch +1:3d}/{epoch} | "
                  f"Loss: {loss:.4f} | "
                  f"Train accuray: {accuracy:.4f}")
            
    return weights, bias, loss_history, accuracy_history



#======================
#4. EVALUATION
#======================

def confusion_matrix_binary(y_true, y_hat):
    true_positive = np.sum((y_true==1) & (y_hat==1))
    true_negative = np.sum((y_true==0) & (y_hat==0))
    false_positive = np.sum((y_true==0) & (y_hat==1))
    false_negative = np.sum((y_true==1) & (y_hat==0))

    return true_positive, true_negative, false_positive, false_negative

def evaluate(X, y, weights, bias, name="Test", save_path=None):
    y_prob = predict_probability(X, weights, bias)
    y_pred = predict(X, weights, bias)
    loss = loss_func(y, y_prob)
    accuracy = accuracy_func(y, y_pred) 
    TP, TN, FP, FN = confusion_matrix_binary(y, y_pred)

    output = (
        f"====== {name} Evaluation ======\n"
        f"Loss     : {loss:.4f}\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"TP = {TP}\n"
        f"TN = {TN}\n"
        f"FP = {FP}\n"
        f"FN = {FN}\n"
        )
    
    print(output)
    
    if save_path is not None:
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(output)

    return {"loss": loss, 
            "accuracy": accuracy,
            "TP": TP, 
            "TN": TN,
            "FP": FP,
            "FN": FN}


def main():
    X_train, y_train, X_test, y_test = load_datasets()

    print("Train shape: ",X_train.shape, y_train.shape)
    print("Test shape: ",X_test.shape, y_test.shape)

    weights, bias, loss_history, accuracy_history = Logistic_Regression_Training(X_train, 
                                                                                 y_train, 
                                                                                 learning_rate=0.12, 
                                                                                 epochs = 200)
    
    evaluate(X_train, y_train, weights, bias, name="Train", save_path="result.txt")
    evaluate(X_test, y_test, weights, bias, name="Test", save_path="result.txt")

    print("\nTRAINING FINISHED!!!")
    print("Final bias = ",bias)
    print("Norm of weights = ", np.linalg.norm(weights))



if __name__ == "__main__":
    main()