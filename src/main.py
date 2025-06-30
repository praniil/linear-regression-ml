import dataset_preprocessing as dp
import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(X, y, learning_rate, epochs):
    m, n = X.shape
    print("m, n", m, n)
    X_b = np.c_[np.ones((m, 1)), X]     #column wise concatenation of 1s in the feature matrix
    w = np.random.randn(n + 1)      #vectorized form of weight. n features n weight and n + 1 for the bias term
    print("w.shape", w.shape)
    print("old weights:", w)

    losses = []

    for i in range(epochs):
        y_predicted = X_b @ w #matrix multiplication
        error = y - y_predicted

        loss = np.mean(error ** 2)
        losses.append(loss)

        gradients = 2 / m * X_b.T @ error
        w -= learning_rate * gradients

        if i % 100 == 0:
            print(f"epochs {i}, Loss: {loss}")
        
    return w, losses

def test_output(X, y, updated_weights, X_mean, X_std):
    X = (X - X_mean) / X_std  # Normalize test input using training stats
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    y_test = X_b @ updated_weights
    print("Prediction:", y_test)
    error = 0.5 * ((y - y_test) ** 2)
    print("Squared Error:", error)
    return y_test


def main():
    # Load and convert to NumPy arrays if they are pandas DataFrames
    X = np.array(dp.X_train)
    y = np.array(dp.y_train)

    # Compute and store normalization stats
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    # Normalize training features
    X = (X - X_mean) / X_std

    # Train model
    updated_weights, losses = gradient_descent(X, y, learning_rate=0.00001, epochs=1000)
    print("\nFinal weights:\n", updated_weights)

    # Predict on a test sample
    test_input = np.array([[1360, 2, 1, 1981, 0.5996366396268326, 0, 5]])
    actual_value = 262382.8522740563
    test_output_value = test_output(test_input, actual_value, updated_weights, X_mean, X_std)

    # Plot and save the loss curve
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error Loss')
    plt.title('Loss Curve During Gradient Descent')
    plt.savefig('loss.png')
    plt.show()

    return updated_weights, losses, test_output_value


if __name__ == "__main__":
    main()