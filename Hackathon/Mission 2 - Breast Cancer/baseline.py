def build_baseline(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    param_grid = {
        'bootstrap': [True],
        'max_depth': [50, 100],
        'max_features': [3, 8],
        'min_samples_leaf': [3, 5],
        'min_samples_split': [2, 5],
        'n_estimators': [300, 500]
    }

    forest_reg = RandomForestClassifier(random_state=42, class_weight="balanced")
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_grid,
                                    n_iter=5, cv=5, scoring='f1_weighted', random_state=42)
    rnd_search.fit(X, y)

    return rnd_search.best_estimator_


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())




