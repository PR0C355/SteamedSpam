def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(X_test)
    Y_test_array = np.array(Y_test)
    print("Overall accuracy of model:", sum(predictions == Y_test_array) / len(predictions))
    print("Accuracy on classifying spam:", sum(np.where(Y_test_array != "ham", predictions == Y_test_array, 0)) / sum(Y_test_array != "ham"))
    print("Accuracy on identifying ham from spam:", sum(np.where(Y_test == "ham", predictions == Y_test, 0)) / sum(Y_test == "ham"))