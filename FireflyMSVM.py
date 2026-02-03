import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import make_scorer, accuracy_score

class FireflyMSVM:
    """
    Firefly Algorithm for Feature Selection with SVM.

    Parameters
    ----------
    n_fireflies : int, default=20
        Number of fireflies.
    max_iterations : int, default=100
        Maximum number of iterations.
    alpha : float, default=0.2
        Randomness parameter (0 to 1).
    beta0 : float, default=1.0
        Attractiveness at distance 0.
    gamma : float, default=1.0
        Absorption coefficient.
    C : float, default=1.0
        Regularization parameter for SVM.
    kernel : str, default='rbf'
        Kernel type for SVM.
    scoring : callable, default=accuracy_score
        Scoring function for evaluating feature subsets.
    """

    def __init__(self, n_fireflies=20, max_iterations=100, alpha=0.2, beta0=1.0,
                 gamma=1.0, C=1.0, kernel='rbf', scoring=accuracy_score):
        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.C = C
        self.kernel = kernel
        self.scoring = scoring

    def _initialize_fireflies(self, n_features):
        return np.random.randint(0, 2, size=(self.n_fireflies, n_features))

    def _calculate_fitness(self, X, y, firefly):
        selected_features = firefly == 1
        if np.sum(selected_features) == 0:  # Avoid empty feature sets.
            return 0.0

        X_selected = X[:, selected_features]
        clf = SVC(C=self.C, kernel=self.kernel)
        scores = cross_val_score(clf, X_selected, y, cv=5, scoring=make_scorer(self.scoring))
        return np.mean(scores)

    def _calculate_attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def _move_firefly(self, firefly_i, firefly_j, attractiveness, n_features):
        new_firefly = firefly_i.copy()
        for k in range(n_features):
            if np.random.rand() < attractiveness:
                new_firefly[k] = firefly_j[k]
            if np.random.rand() < self.alpha:
                new_firefly[k] = 1 - new_firefly[k]  # Randomly flip bits
        return new_firefly

    def fit(self, X, y):
        """
        Fit the Firefly Algorithm for Feature Selection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values (class labels in classification).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        fireflies = self._initialize_fireflies(n_features)
        fitness = np.array([self._calculate_fitness(X, y, firefly) for firefly in fireflies])

        for _ in range(self.max_iterations):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness[j] > fitness[i]:
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        attractiveness = self._calculate_attractiveness(r)
                        fireflies[i] = self._move_firefly(fireflies[i], fireflies[j], attractiveness, n_features)
                        fitness[i] = self._calculate_fitness(X, y, fireflies[i])

        best_firefly = fireflies[np.argmax(fitness)]
        self.selected_features_ = best_firefly == 1
        self.best_fitness_ = np.max(fitness)

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_selected : array-like of shape (n_samples, n_selected_features)
            The input samples with selected features.
        """
        check_array(X)
        if not hasattr(self, 'selected_features_'):
            raise ValueError("FireflySVMFeatureSelection must be fitted before transform.")
        return X[:, self.selected_features_], self.selected_features_

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values (class labels in classification).

        Returns
        -------
        X_selected : array-like of shape (n_samples, n_selected_features)
            The input samples with selected features.
        """
        return self.fit(X, y).transform(X)
