{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_df = pd.read_json('data/final_data.json')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Selecting features and target variable\n",
    "features = ['PERCENT OF HOUSING CROWDED', \n",
    "            'PERCENT HOUSEHOLDS BELOW POVERTY', \n",
    "            'HARDSHIP INDEX',\n",
    "            'PER CAPITA INCOME']\n",
    "\n",
    "target = 'shannon_index'\n",
    "\n",
    "# Splitting the data into training and testing sets\n",
    "X = final_df[features]\n",
    "y = final_df[target]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error (Decision Trees): 13.741316951922567\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeRegressor\n",
    "\n",
    "# Initializing and training the Decision Tree Regressor model\n",
    "tree_model = DecisionTreeRegressor(random_state=42)\n",
    "tree_model.fit(X_train, y_train)\n",
    "\n",
    "# Predicting on the test set\n",
    "y_pred = tree_model.predict(X_test)\n",
    "\n",
    "# Evaluating the model\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(\"Mean Squared Error (Decision Trees):\", mse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error (KNN): 4.415633525599799\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "\n",
    "# Feature scaling\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# Initializing and training the KNN regressor model\n",
    "knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed\n",
    "knn_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Predicting on the test set\n",
    "y_pred_knn = knn_model.predict(X_test_scaled)\n",
    "\n",
    "# Evaluating the model\n",
    "mse_knn = mean_squared_error(y_test, y_pred_knn)\n",
    "print(\"Mean Squared Error (KNN):\", mse_knn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error (Linear Regression): 4.346668043530024\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "model = LinearRegression()\n",
    "model.fit(X_train,y_train)\n",
    "y_pred = model.predict(X_test)\n",
    "mse_linear = mean_squared_error(y_test, y_pred)\n",
    "print(\"Mean Squared Error (Linear Regression):\", mse_linear)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error (Random Forest): 4.290840329025155\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "# Initializing and training the Random Forest classifier\n",
    "rf_classifier = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "rf_classifier.fit(X_train, y_train)\n",
    "\n",
    "# Making predictions on the testing data\n",
    "y_pred = rf_classifier.predict(X_test)\n",
    "\n",
    "# Evaluating the model\n",
    "mse_random = mean_squared_error(y_test, y_pred)\n",
    "print(\"Mean Squared Error (Random Forest):\", mse_random)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# features = ['PERCENT OF HOUSING CROWDED', \n",
    "#             'PERCENT HOUSEHOLDS BELOW POVERTY', \n",
    "#             'PERCENT AGED 16+ UNEMPLOYED', \n",
    "#             'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', \n",
    "#             'PERCENT AGED UNDER 18 OR OVER 64',\n",
    "#             'HARDSHIP INDEX',\n",
    "#             'PER CAPITA INCOME']\n",
    "#Mean Squared Error (Decision Trees): 14.384432122771436\n",
    "#Mean Squared Error (KNN): 4.139463044176787\n",
    "#Mean Squared Error (Linear Regression): 5.303329023877021\n",
    "#Mean Squared Error (Random Forest): 4.368482085574034"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# features = ['PERCENT OF HOUSING CROWDED', \n",
    "#             'PERCENT HOUSEHOLDS BELOW POVERTY', \n",
    "#             'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', \n",
    "#             'PERCENT AGED UNDER 18 OR OVER 64',\n",
    "#             'HARDSHIP INDEX',\n",
    "#             'PER CAPITA INCOME']\n",
    "#Mean Squared Error (Decision Trees): 12.213353087465187\n",
    "#Mean Squared Error (KNN): 4.117839844458724\n",
    "#Mean Squared Error (Linear Regression): 4.763298189159179\n",
    "#Mean Squared Error (Random Forest): 4.2496426590700525"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# features = ['PERCENT OF HOUSING CROWDED', \n",
    "#             'PERCENT HOUSEHOLDS BELOW POVERTY', \n",
    "#             'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', \n",
    "#             'HARDSHIP INDEX',\n",
    "#             'PER CAPITA INCOME']\n",
    "#Mean Squared Error (Decision Trees): 7.32619548931322\n",
    "#Mean Squared Error (KNN): 4.9332521262880284\n",
    "#Mean Squared Error (Linear Regression): 4.4638652250276145\n",
    "#Mean Squared Error (Random Forest): 4.830146615875486"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# features = ['PERCENT OF HOUSING CROWDED', \n",
    "#             'PERCENT HOUSEHOLDS BELOW POVERTY', \n",
    "#             'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', \n",
    "#             'HARDSHIP INDEX',\n",
    "#             'PER CAPITA INCOME']\n",
    "#Mean Squared Error (Decision Trees): 13.741316951922567\n",
    "#Mean Squared Error (KNN): 4.415633525599799\n",
    "#Mean Squared Error (Linear Regression): 4.346668043530402\n",
    "#Mean Squared Error (Random Forest): 4.290840329025155"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
