from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imutils import paths
import argparse
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate


def createImageFeatures(img):
    return img.flatten()


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_Confusion_Matrix(model, X_value, Y_value, title):
    plt.figure(figsize=(8, 5.5))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_value, Y_value, test_size=0.2)

    model.fit(Xtrain, Ytrain)
    display = plot_confusion_matrix(model, Xtest, Ytest, display_labels=["y = -1", "y = 1"], cmap=plt.cm.Greens,
                                    values_format='d')
    Ypredict = model.predict(Xtest)
    print(confusion_matrix(Ytest, Ypredict))

    display.ax_.set_title("Confusion matrix of " + title)
    plt.show()


def plot_KNN_Confusion_Matrix(model, X_value, Y_value, title):
    plt.figure(figsize=(8, 5.5))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_value, Y_value, test_size=0.2)

    model.fit(Xtrain, Ytrain)
    display = plot_confusion_matrix(model, Xtest, Ytest,
                                    display_labels=["y = 1", "y = 2", "y = 3", "y = 4", "y = 5", "y = 6", "y = 7"],
                                    cmap=plt.cm.Greens,
                                    values_format='d')
    Ypredict = model.predict(Xtest)
    print(confusion_matrix(Ytest, Ypredict))

    display.ax_.set_title("Confusion matrix of " + title)
    plt.show()


from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


def plot_KNN_F1_Score(K_range, X, y):
    mean_error, std_error = [], []
    cv = KFold(n_splits=5, shuffle=False)
    for k in K_range:
        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index[0]:train_index[-1] + 1], X[test_index[0]:test_index[-1] + 1]
            y_train, y_test = y[train_index[0]:train_index[-1] + 1], y[test_index[0]:test_index[-1] + 1]
            model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average=None)
            mean_error.append(np.array(score).mean())
            std_error.append(np.array(score).std())
            print('在训练')
    plt.figure(figsize=(8, 5.5))
    plt.errorbar(K_range, mean_error, yerr=std_error, linewidth=2,
                 label="F1 score and standard deviation / K = " + str(k))
    plt.gca().set(xlabel='K', ylabel="F1 score")
    plt.legend(loc='lower right')
    plt.show()


def get_best_p_and_k(K_range, X, y):
    param_grid = {"n_neighbors": K_range, "p": [1, 2]}
    model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
    history = model.fit(X, y)
    print(model.best_params_)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', help='train image directory', default='./output_dir/data_knn', type=str)
    parser.add_argument('--val-dir', help='validation image directory', default='./output')
    args = parser.parse_args()

    image_paths = list(paths.list_images(args.train_dir))
    X = []
    y = []

    for (i, image_path) in enumerate(image_paths):
        image = cv2.imread(image_path)
        label = image_path.split('/')[-1].split('_')[0]
        pixels = createImageFeatures(image)
        X.append(pixels)
        y.append(label)

    X = np.array(X)
    y = np.array(y)


    (X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.1, random_state=0)

    model = KNeighborsClassifier(n_neighbors=25, weights='uniform')
    plot_KNN_Confusion_Matrix(model, X, y, 'KNN Confusion Matrix with K = 25')


    k_range = [1, 5, 9, 17, 25]
    # plot_KNN_F1_Score(k_range, X, y)
    knn_model = get_best_p_and_k(k_range, X_train, Y_train)
    Y_pre = knn_model.best_estimator_.predict(X_test)
    accuracy = np.count_nonzero((Y_pre == Y_test) == True) / len(Y_test)
    print("Prediction accuracy is:", accuracy)

    from sklearn.dummy import DummyClassifier

    dummy_model = DummyClassifier(strategy='most_frequent').fit(X_train, Y_train)
    plot_Confusion_Matrix(dummy_model, X, y, 'Baseline Classification')


if __name__ == '__main__':
    main()
