from packages import *
# importing csv file
diabetes_data = pd.read_csv(r'C:\Users\Aju Pradhan\Desktop\all_together_project\diabetes.csv')
################################################################
X = diabetes_data.drop("Outcome", axis=1).values
Y = diabetes_data["Outcome"]
################################################################
scalar = StandardScaler()
scalar.fit(X)
###############################
standardised_data = scalar.transform(X)
X = standardised_data
Y=diabetes_data["Outcome"]
#Train test splitting
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
