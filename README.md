✈️ Flight Delay Prediction and Optimization
📘 Overview

This project focuses on predicting and minimizing flight delays using a combination of machine learning models and prescriptive optimization techniques.
The workflow follows the three pillars of analytics:

Diagnostic Analytics: Understanding delay causes and patterns.

Predictive Analytics: Forecasting delay probabilities using ML models.

Prescriptive Analytics: Recommending actions to minimize expected delays through optimization.

🎯 Project Objectives

Clean and preprocess large flight datasets.

Engineer meaningful features such as time of day, season, and route frequency.

Build and compare multiple predictive models for flight delay classification.

Implement a linear programming model to minimize total delay time.

Visualize insights to help airport authorities make data-driven decisions.

⚙️ Tech Stack

Programming Language: Python

Libraries:

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn, XGBoost

Optimization: PuLP
🧠 Methodology
1️⃣ Data Preprocessing

Handled missing values:

Numerical features: Filled using the median.

Categorical features: Filled using the mode.

Removed columns with excessive missing values (e.g., CANCELLATION_CODE).

Saved clean dataset for modeling.

2️⃣ Feature Engineering

Extracted time-based features: year, month, day, day of week, quarter.

Created categorical features:

IS_WEEKEND, SEASON, and TIME_OF_DAY.

Built route-based metrics like route frequency and airport daily traffic.

Defined target variable (IS_DELAYED) where flights delayed > 15 mins are labeled as 1.

3️⃣ Predictive Analytics

Implemented four ML models:

Model	Accuracy	F1 Score	ROC AUC
Logistic Regression	0.9939	0.9827	1.0000
Random Forest	0.9969	0.9912	0.9997
XGBoost	0.9994	0.9982	1.0000
Neural Network	1.0000	0.9999	1.0000

Best Model: Neural Network (highest overall performance).

4️⃣ Prescriptive Analytics

Used Linear Programming (PuLP) to optimize rescheduling decisions.

Constraint: reschedule a maximum of 50 flights.

Achieved total expected delay reduction of 517,413.3 minutes.

Generated insights and recommendations for airport scheduling improvements.

📊 Key Insights

Departure delay is highly correlated with arrival delay.

Peak delay periods occur during afternoon and evening hours.

Weather, carrier issues, and late aircraft arrivals are major delay contributors.

Machine learning models can accurately predict potential flight delays before departure.

💡 Recommendations for Airport Authorities

Prioritize rescheduling of flights with delay probability > 0.7.

Allocate additional ground resources during peak delay hours (afternoon/evening).

Monitor routes and carriers with frequent delay patterns for operational improvements.

Use predictive models for real-time scheduling and proactive passenger communication.

🚀 How to Run

Clone the repository

git clone https://github.com/<your-username>/flight-delay-prediction.git
cd flight-delay-prediction


Install dependencies

pip install -r requirements.txt


Run Jupyter Notebook

jupyter notebook


Open the notebooks in the /notebooks folder and execute step by step.

👨‍💻 Author

Kavindu Lakshan
📧 lakshankasthuriarachchi@gmail.com
