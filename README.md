# GCU_ML_PHW1
## 1. Objective Setting
We would like to create a “single major function” that will automatically run different combinations of the following : Data Scaling, Data Encoding, Classification Models with different K parameters, and various number k for k-fold cross validation.
We will use the Wisconsin Cancer Dataset to test the algorithm in the end-to-end process as used in Data Science.
The Program will run through loading data, data preparation, data analysis, data inspection automatically and show the best 5 results of combination.

## 2. Data Curation
Dataset Name : Wisconsin Cancer Dataset
Source : https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/ 
	(breast-cancer-wisconsin.data)

## 3. Data Inspection
- Number of Instances: 699 (as of 15 July 1992)
- Number of Attributes: 10 plus the class attribute
- Attribute Information: (class attribute has been moved to last column)
   #  Attribute                     Domain
   --------------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
- Missing attribute values: 16
   There are 16 instances in Groups 1 to 6 that contain a single missing 
   (i.e., unavailable) attribute value, now denoted by "?".  
   
## 4. Data Preparation
Dirty data cleaning
Bare nuclei is the only attribute with missing values, ‘?’.
→ we replaced ‘?’ to mode value.
Feature engineering
Since the first attribute ‘sample code number’ is useless in analyzing process, we will drop this attribute. Every other attributes are necessary

 Scaling (5) - Standard, MinMax, Robust, MaxAbsScaler, Normalizer
 Encoding (0) – All data are numerical numbers, so no Encoding is needed.


## 5. Combination cases
Scaling(5) * Encoding(-) * Algorithms(4) * K fold(3) = 60
We plan to make 60 cases.

## 6. Algorithm Structure
scaler = [‘StandardScaler()’, ‘MinMaxScaler()’, ‘RobustScaler()’, ‘MaxAbsScaler()’, ‘Normalizer()’]
model = [‘DecisionTreeClassifier(criterion=’gini)’, ‘DecisionTreeClassifier(criterion=’entrophy)’, …]
def best_comb(scaler, model):
    """Train model and find best combination of classifier and scaler

    Args:
        scaler: Array of scaler you want to use
        model: Array of classifier model you want to use

    Returns:
        best_acc: return best accuracy among combination of scaler and model
        best_scaler: return scaler which has best score
        best_model: return model which has best score
        x_bTest: return x_test which has best score
        y_Btest: return y_test which has best score
        y_Bpred: return y_pred which has best score
    """
    best_acc = 0
    
    for element in scaler:
        scaler = eval(element)
        scaled = scaler.fit_transform(X_clf)
        x_train, x_test, y_train, y_test = train_test_split(scaled, y_clf, test_size = 0.2, random_state=42) #Using random_state to fixing random rate
        for element2 in model:
            classifier = eval(element2)
            classifier = classifier.fit(x_train,y_train)
            
            y_pred = classifier.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print(f'Using {classifier} in {scaler} score : {acc}')

            if acc > best_acc:
                best_acc = acc
                best_scaler = element
                best_model = classifier
                x_bTest = x_test
                y_bTest = y_test
                y_bPred = y_pred
        print('')
    return best_acc, best_scaler, best_model, x_bTest, y_bTest, y_bPred

and calculate k-fold validation

