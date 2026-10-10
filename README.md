# SpaceX Falcon 9 Landing Prediction — IBM Data Science Capstone

Predicting whether the Falcon 9 first stage will land successfully, using data collected from the SpaceX REST API and Wikipedia, explored with SQL and visualizations, mapped with Folium, and modeled with several classification algorithms.

Capstone project for the [IBM Data Science Professional Certificate](https://www.coursera.org/account/accomplishments/professional-cert/1WQT728YKSJ5) (Coursera, August 2026). The notebooks are the IBM Skills Network lab templates, completed by me for the capstone; IBM's copyright headers are kept as issued.

## Notebooks

| Notebook | Description |
|---|---|
| [`jupyter-labs-spacex-data-collection-api.ipynb`](jupyter-labs-spacex-data-collection-api.ipynb) | Collects launch data from the SpaceX REST API and cleans it into `dataset_part_1.csv`. The API is now offline, so outputs after the per-launch lookups are cleared (explained in the notebook) |
| [`jupyter-labs-webscraping.ipynb`](jupyter-labs-webscraping.ipynb) | Scrapes Falcon 9 launch records from Wikipedia with BeautifulSoup |
| [`labs-jupyter-spacex-Data wrangling-v2.ipynb`](labs-jupyter-spacex-Data%20wrangling-v2.ipynb) | Exploratory data analysis and landing outcome labeling, producing `dataset_part_2.csv` |
| [`jupyter-labs-eda-sql-coursera_sqllite.ipynb`](jupyter-labs-eda-sql-coursera_sqllite.ipynb) | Exploratory data analysis using SQL (SQLite) |
| [`jupyter-labs-eda-dataviz-v2.ipynb`](jupyter-labs-eda-dataviz-v2.ipynb) | Exploratory data analysis and feature engineering with visualizations, producing `dataset_part_3.csv` |
| [`lab-jupyter-launch-site-location-v2.ipynb`](lab-jupyter-launch-site-location-v2.ipynb) | Interactive launch site mapping and proximity analysis with Folium |
| [`SpaceX-Machine-Learning-Prediction-Part-5-v1.ipynb`](SpaceX-Machine-Learning-Prediction-Part-5-v1.ipynb) | Builds and compares Logistic Regression, SVM, Decision Tree, and KNN classifiers to predict landing success |

[`spacex_dash_app.py`](spacex_dash_app.py) is a Plotly Dash dashboard of launch success by site and payload mass, reading `spacex_launch_dash.csv`. Run it with `pip install pandas dash plotly` and `python spacex_dash_app.py`.

## Results

90 Falcon 9 launches remain after removing Falcon 1 launches and multi-core/multi-payload records, with an overall first-stage landing success rate of 66.7%.

| Model | 10-fold CV accuracy (72 training launches) | Test accuracy (18 launches) |
|---|---|---|
| Decision Tree | 86.3% | 83.3% |
| SVM | 84.8% | 83.3% |
| KNN | 84.8% | 83.3% |
| Logistic Regression | 84.6% | 83.3% |

The test set does not separate the models: all four classify 15 of 18 launches correctly. The decision tree's cross-validation lead is under two points, and it was chosen from a 324-configuration grid against three for logistic regression, so part of that lead is selection on the same folds rather than a better model. With 90 launches, these numbers support "landing outcome is predictable at roughly 83–86%", not a ranking.

## Changes from the lab template

The machine learning notebook was re-run with current scikit-learn, which required two changes to the lab's parameter grids:

- `max_features='auto'` was removed from the decision tree grid. scikit-learn 1.3 dropped that option (for classifiers it meant `'sqrt'`, which stays in the grid), and it was making half of the grid's fits fail.
- The deprecated `penalty=['l2']` was removed from the logistic regression grid; `l2` is the default.

The decision tree is also seeded (`random_state=0`) so its results reproduce from run to run.
