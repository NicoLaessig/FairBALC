"""
This code coordinates the training of each classifier and calls the corresponding training
functions to be executed.
"""
import joblib


class ModelOps():
    """This class calls all classifiers to train their models.

    Parameters
    ----------
    model_dict: dictionary
        Saves information about each trained classifier.

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.
    """
    def __init__(self, model_dict, iteration):
        self.model_dict = model_dict
        self.iteration = iteration


    def return_dict(self):
        """
        Returns the model dictionary.
        """
        return self.model_dict


    def run(self, model_obj, model, folder, input_file=None, sbt=False, attrs=None):
        """Takes as input the model that will be trained and will return the trained model
        name and will save the model as .pkl & also save some informations in the dictionary.

        Parameter
        -------
        model_obj: Object
            Instance of the Model class.

        model: str
            Name of the classifier that should be trained and saved.

        folder: str
            String of the folder location + prefix.

        input_file: str
            Name of the dataset.

        sbt: boolean
            Set to TRUE if split before training is activated.

        attrs: list
            List contains parameter values if classic AdaBoost or classic RandomForest
            approach is chosen.


        Returns
        -------
        joblib_file: str
            Name of the .pkl file of the trained classifier.
        """
        if model == "DiversePool":
            classifier_list, prediction_list, model_name, pool = model_obj.diverse_ensemble(
                target_pool_size=30,
                accuracy_weight = 0.2,
                fairness_weight = 0.6,
                diversity_weight = 0.2,
                include_egr = False,
                include_gsr = False,
                include_adafair = False,
                include_subgroup = False
                )
            joblist_file_list = []
            with open(folder + 'diverse_pool.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
    
        elif model == "DecisionTree":
            classifier, prediction, model_name = model_obj.decision_tree()
        elif model == "LinearSVM":
            classifier, prediction, model_name = model_obj.linear_svm()
        elif model == "NonlinearSVM":
            classifier, prediction, model_name = model_obj.nonlinear_svm()
        elif model == "LogisticRegression":
            classifier, prediction, model_name = model_obj.log_regr()
        elif model == "SoftmaxRegression":
            classifier, prediction, model_name = model_obj.softmax_regr()
        elif model == "LFR":
            classifier, prediction, model_name = model_obj.lfr360()
        elif model == "DisparateImpactRemover":
            classifier, prediction, model_name = model_obj.dir360()
        elif model == "Reweighing":
            classifier, prediction, model_name = model_obj.rew360()
        elif model == "GridSearchReduction":
            classifier, prediction, model_name = model_obj.gsr360()
        elif model == "GerryFairClassifier":
            classifier, prediction, model_name = model_obj.gfc360()
        elif model == "CalibratedEqOddsPostprocessing":
            classifier, prediction, model_name = model_obj.ceop360()
        elif model == "EqOddsPostprocessing":
            classifier, prediction, model_name = model_obj.eop360()
        elif model == "RejectOptionClassification":
            classifier, prediction, model_name = model_obj.roc360()
        elif model == "FaX":
            classifier, prediction, model_name = model_obj.fax360()
        elif model == "Fair-SMOTE":
            classifier, prediction, model_name = model_obj.smote360()
        elif model == "LTDD":
            classifier, prediction, model_name = model_obj.ltdd360()
        elif model == "FairRR":
            classifier, prediction, model_name = model_obj.fairrr360()
        elif model == "FairSSL-Lx":
            classifier, prediction, model_name = model_obj.ssllx360()
        elif model == "FairSSL-xT":
            classifier, prediction, model_name = model_obj.sslxt360()
        elif model == "FAGTB":
            classifier, prediction, model_name = model_obj.fagtb360()
        elif model == "JiangNachum":
            classifier, prediction, model_name = model_obj.jn360()
        elif model == "AdaFair360":
            classifier, prediction, model_name = model_obj.adafair360()
        elif model == "FairnessConstraintModel":
            classifier, prediction, model_name = model_obj.fcm360()
        elif model == "HSICLinearRegression":
            classifier, prediction, model_name = model_obj.hsic360()
        elif model == "GeneralFairERM":
            classifier, prediction, model_name = model_obj.gferm360()
        elif model == "FairBoost360":
            classifier, prediction, model_name = model_obj.fairboost360()
        elif model == "fairret":
            classifier, prediction, model_name = model_obj.fairret360()
        elif model == "FESF":
            classifier, prediction, model_name = model_obj.fesf360()
        elif model == "GradualCompatibility":
            classifier, prediction, model_name = model_obj.gc360()
        elif model == "FairHeckman":
            classifier, prediction, model_name = model_obj.heckman_reg()
        elif model == "ConvexFrameworkModel_reg":
            classifier, prediction, model_name = model_obj.cfm_reg()
        elif model == "HSICLinearRegression_reg":
            classifier, prediction, model_name = model_obj.hsic_reg()
        elif model == "GeneralFairERM_reg":
            classifier, prediction, model_name = model_obj.gferm_reg()
        elif model == "FairGeneralizedLinearModel_reg":
            classifier, prediction, model_name = model_obj.fglm_reg()
        elif model == "FairDummies_reg":
            classifier, prediction, model_name = model_obj.fd_reg()
        elif model == "HGR_reg":
            classifier, prediction, model_name = model_obj.hgr_reg()
        elif model == "UnawareFairReg":
            classifier, prediction, model_name = model_obj.unaware_reg()
        elif model == "ReductionApproach":
            classifier, prediction, model_name = model_obj.reduction_reg()
        elif model == "AdversarialDebiasing_reg":
            classifier, prediction, model_name = model_obj.adv_deb_reg()
        elif model == "AdaptedAdaBoost":
            classifier_list, prediction_list, model_name = model_obj.adapted_adaboost()
            joblist_file_list = []
            with open(folder + 'adaboost.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "RandomForestClassic":
            classifier_list, prediction_list, model_name, rfc = model_obj.rf_classic(n_estimators=attrs[1],
                max_depth=attrs[2], criterion=attrs[3])
            joblib_file = folder + "RandomForestClassic" + str(attrs[0])  + ".pkl"
            joblib.dump(rfc, joblib_file)
            joblist_file_list = []
            with open(folder + 'rf_classic.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "AdaBoostClassic":
            classifier_list, prediction_list, model_name, abc = model_obj.adaboost_classic()
            joblib_file = folder + "AdaBoostClassic.pkl"
            joblib.dump(abc, joblib_file)
            joblist_file_list = []
            with open(folder + 'adaboost_classic.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "ExtraTrees":
            classifier_list, prediction_list, model_name, abc = model_obj.extra_trees(large=False)
            joblist_file_list = []
            with open(folder + 'extra_trees.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "ExtraTrees2":
            classifier_list, prediction_list, model_name, abc = model_obj.extra_trees(large=True)
            joblist_file_list = []
            with open(folder + 'extra_trees.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "RandomForest2":
            classifier_list, prediction_list, model_name, abc = model_obj.rf2(large=True)
            joblist_file_list = []
            with open(folder + 'random_forest.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "AdaBoost2":
            classifier_list, prediction_list, model_name, abc = model_obj.ada2(large=True)
            joblist_file_list = []
            with open(folder + 'ada_boost.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "BoostReg":
            classifier_list, prediction_list, model_name, abc = model_obj.boost_reg()
            joblist_file_list = []
            with open(folder + 'BoostReg.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "XGBoost2":
            classifier_list, prediction_list, model_name, abc = model_obj.xgb2(large=True)
            joblist_file_list = []
            with open(folder + 'xgb_boost.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        
        elif model in ["OptimizedRandomForest", "OptimizedAdaBoost", "OptimizedXGBoost", "OptimizedExtraTrees"]:
            if model == "OptimizedRandomForest":
                classifier_list, prediction_list, model_name = model_obj.opt_learner("RandomForest", input_file, sbt)
                joblist_file_list = []
                with open(folder + 'optimized_random_forest.txt', 'w') as f:
                    f.write(str(classifier_list))
            elif model == "OptimizedAdaBoost":
                classifier_list, prediction_list, model_name = model_obj.opt_learner("AdaBoost", input_file, sbt)
                joblist_file_list = []
                with open(folder + 'optimized_adaboost.txt', 'w') as f:
                    f.write(str(classifier_list))
            elif model == "OptimizedXGBoost":
                classifier_list, prediction_list, model_name = model_obj.opt_learner("XGBoost", input_file, sbt)
                joblist_file_list = []
                with open(folder + 'optimized_xgbboost.txt', 'w') as f:
                    f.write(str(classifier_list))
            elif model == "OptimizedExtraTrees":
                classifier_list, prediction_list, model_name = model_obj.opt_learner("ExtraTrees", input_file, sbt)
                joblist_file_list = []
                with open(folder + 'optimized_extratrees.txt', 'w') as f:
                    f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "AdaFair":
            classifier_list, prediction_list, model_name, af = model_obj.adafair()
            joblist_file_list = []
            with open(folder + 'adafair.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "FairBoost":
            classifier_list, prediction_list, model_name, fb = model_obj.fairboost()
            joblist_file_list = []
            with open(folder + 'fairboost.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        
        elif model == "EGR":
            classifier_list, prediction_list, model_name, af = model_obj.egr()
            joblist_file_list = []
            with open(folder + 'egr.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        
        elif model == "EGR2":
            classifier_list, prediction_list, model_name, af = model_obj.egr2()
            joblist_file_list = []
            with open(folder + 'egr.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        
        elif model == "EGR_reg":
            classifier_list, prediction_list, model_name, af = model_obj.egr_reg()
            joblist_file_list = []
            with open(folder + 'egr_reg.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        
        elif model == "GSR":
            classifier_list, prediction_list, model_name, af = model_obj.gsr()
            joblist_file_list = []
            with open(folder + 'gsr.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list

        elif model == "GSR2":
            classifier_list, prediction_list, model_name, af = model_obj.gsr(20)
            joblist_file_list = []
            with open(folder + 'gsr.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list

        elif model == "LinearRegression":
            classifier, prediction, model_name = model_obj.linear_regression()
        elif model == "SGDRegressor":
            classifier, prediction, model_name = model_obj.sgd_regressor()
        elif model == "Ridge":
            classifier, prediction, model_name = model_obj.ridge()
        elif model == "ElasticNet":
            classifier, prediction, model_name = model_obj.elasticnet()
        elif model == "Lasso":
            classifier, prediction, model_name = model_obj.lasso()
        elif model == "Lars":
            classifier, prediction, model_name = model_obj.lars()
        elif model == "BayesianRidge":
            classifier, prediction, model_name = model_obj.bayesian_ridge()
        elif model == "HuberRegressor":
            classifier, prediction, model_name = model_obj.huber_regressor()

        elif model == "LogisticRegression_mult":
            classifier, prediction, model_name = model_obj.log_regr_mult()
        elif model == "SVM_mult":
            classifier, prediction, model_name = model_obj.svm_mult()
        elif model == "RandomForest_mult":
            classifier, prediction, model_name = model_obj.random_forest_mult()
        elif model == "XGBoost_mult":
            classifier, prediction, model_name = model_obj.xgboost_mult()
        elif model == "LightGBM_mult":
            classifier, prediction, model_name = model_obj.lightgbm_mult()
        elif model == "kNN_mult":
            classifier, prediction, model_name = model_obj.kNN_mult()
        elif model == "NaiveBayes_mult":
            classifier, prediction, model_name = model_obj.naive_bayes_mult()
        elif model == "NN_mult":
            classifier, prediction, model_name = model_obj.nn_mult()
        elif model == "DecisionTree_mult":
            classifier, prediction, model_name = model_obj.dectree_mult()

        
        d_list = []
        joblib_file = folder + model_name + "_model.pkl"
        joblib.dump(classifier, joblib_file)
        d_list.append(joblib_file)
        d_list.append(prediction)
        #Dictionary containing all models of the following form: {Model Name: [(1) Saved Model
        #as .pkl, (2) Prediction of the model for our test data]
        #Train and save each model on the training data set.
        self.model_dict[joblib_file] = d_list

        return joblib_file
