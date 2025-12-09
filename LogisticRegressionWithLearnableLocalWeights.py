import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc

class LogisticRegressionWithLearnableLocalWeights:
    def __init__(self, lr_theta=0.01, lr_lambda=0.0001, num_iter=100000, fit_intercept=True, verbose=False, validate_w_auc = True, test_auc_threshold = 0.99):
        self.lr_theta = lr_theta
        self.lr_lambda = lr_lambda
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.validate_w_auc = validate_w_auc
        self.eplison = 0.000001
        self.test_auc_threshold = test_auc_threshold
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (- self.class_weight* y * np.log(h+self.eplison) - (1 - y) * np.log(1 - h+self.eplison)).mean()
        
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            pred1 = self.__sigmoid(np.dot(X,self.theta))

        
        return np.stack((1-pred1,pred1),axis=1) 
        
    def predict(self, X, cut_off):
        return np.where(self.predict_proba(X) >= cut_off, 1, 0)
    
    def fit(self, X, y, test_set_x, test_set_y):
        self.xb_trace = []
        self.lambda_trace = []
        self.beta_trace = []
        self.precision_trace = []
        self.iterations = 0

        if self.fit_intercept:
            X = self.__add_intercept(X)
            test_set_x = self.__add_intercept(test_set_x)
        
        # weights initialization
        theta = np.ones(X.shape[1])
        self.class_weight = np.full((len(y),), 1.0) #use (len(y),) to denote np array.  Need the comma in the tuple. 
        
        test_auc_list = []
        winner_theta = 0.0
        winner_test_auc = 0.0
        
        for i in range(self.num_iter):
            z = np.dot(X, theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (-self.class_weight*y*(1-h) + (1-y)*h)) / y.size
            gradient_lambda = y * np.log(h)
            #gradient_lambda = -1/self.lr_lambda*y*(1/(self.class_weight +1/self.lr_lambda))
            theta -= self.lr_theta * gradient
            self.class_weight -= self.lr_lambda * gradient_lambda   
            
            if np.any(self.class_weight<0):
                break
            
            z = np.dot(X, theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            
            if self.verbose==True:
                self.lambda_trace.append(self.class_weight[y > 0]) 
                self.xb_trace.append(z[y > 0])
                self.beta_trace.append(theta.copy())

            if i%1000==0: 
                # if i%5000 == 0:
                #     print("Iteration: " + str(i))
                #     print("Theta: {}".format(theta))
                #     print('mean of local weights: %f' %(self.class_weight[self.class_weight > 1].mean()))
                probas_ = self.__sigmoid(np.dot(test_set_x, theta))

                if self.validate_w_auc:
                    fpr, tpr, _ = roc_curve(test_set_y, probas_)
                    test_auc = round(auc(fpr, tpr), 4) 
                else: 
                    test_auc = round(average_precision_score(test_set_y, probas_), 4)
                # For imbalanced classification with a severe skew and few examples of the minority class, the ROC AUC can be misleading. This is because a small number of correct or incorrect predictions can result in a large change in the ROC Curve or ROC AUC score. https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/
                #from sklearn.metrics import precision_recall_curve
                #prcision,recall,_ = precision_crall_curve(y_test, pos_probs)
                #CHANGE THIS TO PRECISION RECALL AUC auc_score = auc(recall,precision)
                    
                if i == 0:
                    test_auc_list.append(test_auc)
                
                test_auc_list.append(test_auc)
                self.precision_trace.append(test_auc)

                #Only captures new theta if precision score is improved.
                if test_auc >= max(test_auc_list[:-1]): 
                    winner_test_auc = test_auc
                    winner_theta = list(theta)

                #after a decrease/no change in precision after 20000 straight iterations, break
                # if len(test_auc_list) > 20 & sum(test_auc <= k for k in test_auc_list[-20:]) == 20:
                #      print("Break at iteration {0:d} due to decrease or no change in average precision for 20,000 iterations".format(i))
                #      print(test_auc_list)
                #      break
                
                # if self.validate_w_auc: 
                #     print("AUC: " + str(test_auc))
                # else:
                #     print("The Average Precision Score is: " + str(test_auc))
                
                if test_auc <= self.test_auc_threshold:
                    #print('Continue')
                    #print(theta)
                    continue
                else:
                    break
                    
        self.iterations = i 
        self.theta = list(winner_theta)