import copy
import math
import itertools
import joblib
import ast
import shelve
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from algorithm.FALCCClassic_files import RunTraining
from algorithm.FALCCAddons import Metrics, Metrics_Reg, Metrics_Multi
from algorithm.FALCCAddons.parameter_estimation import log_means


class FALCC:
    """This class calls runs the 3rd step and online phase of the FALCC algorithm.

    Parameters
    ----------
    link: str
        Link of the output directory.

    filename: str
        Name of the dataset (used for naming).

    df: {array-like, sparse matrix}, shape (n_samples, m_features)
        Whole DataFrame

    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    favored: tuple of float
        Tuple of the values of the favored group.

    label: string
        String name of the target column.

    training: string
        String name of which training procedure is chosen

    proxy: string
        Name of the proxy strategy used

    metric: string
        Name of the metric which should be used to get the best result.

    lam: float (0-1)
        Value to balance the accuracy and fairness parts of the metrics.
        Under 0.5: Give fairness higher importance.
        Over 0.5: Give accuracy higher importance.

    cluster_algorithm: string
        String of the parameter estimation algorithm that should be chosen.

    ccr: List of size 2 of integers
        Minimum and maximum amount of clusters that should be generated.
        Unbounded is defined by -1.

    trained_models: list of strings
        Location of already trained classifiers (they have to be in .pkl format)

    allowed: list of strings
        Feature names that should not be affected by the rpoxy mitigation strategy

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.

    sbt: boolean
        Value is set to true if the classifiers should only be trained on subset of the data
        or on the whole dataset.
        IT HAS TO BE SET TO FALSE IN THE CURRENT VERSION.

    """
    def __init__(self, link, df, input_file, df_dict, metric, lam=0.5, training="opt_adaboost",
        validationsize=0.4, proxy="no", cluster_algorithm="LOGmeans", ccr=[-1,-1], cluster_pre=False,
        trained_models=None, allowed="", remove=False, reg=True, sbt=False, iteration=-1):
        self.link = link
        self.df = df
        self.input_file = input_file
        self.df_dict = df_dict
        self.metric = metric
        self.df_dict["metric"] = re.sub("bea_", "", metric)
        self.lam = lam
        self.training = training
        self.validationsize = validationsize
        self.proxy = proxy
        self.cluster_algorithm = cluster_algorithm
        self.ccr = ast.literal_eval(ccr)
        self.cluster_pre = cluster_pre
        self.trained_models = trained_models
        self.allowed = allowed
        self.remove = remove
        self.reg = reg
        self.sbt = sbt
        self.iteration = iteration


    def fit(self, X_train, y_train):
        """The offline phase of the FALCC algorithm.

        Parameters
        ----------
        X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
            Training data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_train: array-like, shape (n_samples)
            Label vector relative to the training data X_train.
        """
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validationsize, random_state=42)

        if not self.reg:
            classes = len(self.df[self.df_dict["label"]].unique())
            if classes > 2:
                model_training_list = ["LogisticRegression_mult", "SVM_mult", "RandomForest_mult",\
                    "kNN_mult", "NaiveBayes_mult", "NN_mult", "DecisionTree_mult"]
            else:
                if self.training == "single_classifiers":
                    model_training_list = ["DecisionTree", "LinearSVM", "NonlinearSVM",\
                        "LogisticRegression", "SoftmaxRegression"]
                #This option requires some additional work to implement FaX, Fair-SMOTE and LFR in FALCC
                #Removed the option in this version: Instead use the general FALCC approach
                #elif self.training == "fair":
                 #   model_training_list = ["FaX", "Fair-SMOTE", "LFR"]
                elif self.training == "opt_random_forest":
                    model_training_list = ["OptimizedRandomForest"]
                elif self.training == "opt_adaboost":
                    model_training_list = ["OptimizedAdaBoost"]
                elif self.training == "adapted_adaboost":
                    model_training_list = ["AdaptedAdaBoost"]
                elif self.training == "adafair":
                    model_training_list = ["AdaFair"]
                elif self.training == "fairboost":
                    model_training_list = ["FairBoost"]
                elif self.training == "ET":
                    model_training_list = ["ExtraTrees"]
                elif self.training == "EGR":
                    model_training_list = ["EGR"]
                elif self.training == "EGRGSR":
                    model_training_list = ["EGR", "GSR"]

            index = self.df.index.name
            train_id_list = []
            for i, row in X_train.iterrows():
                train_id_list.append(i)
            #y_train = y_train.to_frame()

            val_id_list = []
            for i, row in X_val.iterrows():
                val_id_list.append(i)
            #y_val = y_val.to_frame()

            if self.training != "no":
                run_main = RunTraining(X_val, y_val, val_id_list, self.df_dict["sens_attrs"],
                    index, self.df_dict["label"], self.df_dict["favored"], self.link, self.input_file,
                    self.iteration, False, self.remove, self.df_dict)

                val_df, d, model_list, model_comb = run_main.train(model_training_list, X_train, y_train,
                    [])
                val_df.to_csv(self.link + "test_predictions.csv", index_label=index)
                val_df = val_df.sort_index()
            else:
                d = dict()
                model_list = []
                val_df = pd.DataFrame(columns=[index, self.df_dict["label"]])
                val_df[index] = list(y_val.index)
                val_df[self.df_dict["label"]] = y_val[self.df_dict["label"]]
                for sens in self.df_dict["sens_attrs"]:
                    val_df[sens] = X_val[sens]
                for tm in self.trained_models:
                    used_model = joblib.load(tm)
                    prediction = used_model.predict(X_val)
                    val_df[tm] = prediction
                    model_list.append(tm)
                    d_list = []
                    d_list.append(tm)
                    d_list.append(prediction)
                    d[tm] = d_list
                groups = len(self.df.groupby(self.df_dict["sens_attrs"]))
                model_comb = list(itertools.product(model_list, repeat=groups))

                val_df.to_csv(self.link + "test_predictions.csv", index_label=index)
                val_df = val_df.sort_index()

        else:
            model_training_list = ["LinearRegression", "SGDRegressor", "Ridge", "Lasso", "ElasticNet", "BayesianRidge", "HuberRegressor", "Lars"]

            index = self.df.index.name
            train_id_list = []
            for i, row in X_train.iterrows():
                train_id_list.append(i)
            #y_train = y_train.to_frame()

            val_id_list = []
            for i, row in X_val.iterrows():
                val_id_list.append(i)
            #y_val = y_val.to_frame()

            if self.training != "no":
                run_main = RunTraining(X_val, y_val, val_id_list, self.df_dict["sens_attrs"],
                    index, self.df_dict["label"], self.df_dict["favored"], self.link, self.df,
                    False, self.remove, self.df_dict)

                val_df, d, model_list, model_comb = run_main.train(model_training_list, X_train, y_train,
                    [])
                val_df.to_csv(self.link + "test_predictions.csv", index_label=index)
                val_df = val_df.sort_index()
            else:
                d = dict()
                model_list = []
                val_df = pd.DataFrame(columns=[index, self.df_dict["label"]])
                val_df[index] = list(y_val.index)
                val_df[self.df_dict["label"]] = y_val[self.df_dict["label"]]
                for sens in self.df_dict["sens_attrs"]:
                    val_df[sens] = X_val[sens]
                for tm in self.trained_models:
                    used_model = joblib.load(tm)
                    prediction = used_model.predict(X_val)
                    val_df[tm] = prediction
                    model_list.append(tm)
                    d_list = []
                    d_list.append(tm)
                    d_list.append(prediction)
                    d[tm] = d_list
                groups = len(self.df.groupby(self.df_dict["sens_attrs"]))
                model_comb = list(itertools.product(model_list, repeat=groups))

                val_df.to_csv(self.link + "test_predictions.csv", index_label=index)
                val_df = val_df.sort_index()

        #Estimate the clustersize and then create the clusters
        if self.cluster_pre == False:
            X_val_new = copy.deepcopy(X_val)
            if self.proxy == "reweigh":
                with open(self.link + "reweighing_attributes.txt", 'w') as outfile:
                    df_new = copy.deepcopy(self.df)
                    self.weight_dict = dict()
                    cols = list(df_new.columns)
                    cols.remove(self.df_dict["label"])
                    for sens in self.df_dict["sens_attrs"]:
                        cols.remove(sens)

                    for col in cols:
                        if col in self.allowed:
                            self.weight_dict[col] = 1
                            continue
                        x_arr = df_new[col].to_numpy()
                        col_diff = 0
                        for sens in self.df_dict["sens_attrs"]:
                            z_arr = df_new[sens]
                            sens_corr = abs(pearsonr(x_arr, z_arr)[0])
                            if math.isnan(sens_corr):
                                sens_corr = 1
                            col_diff += (1 - sens_corr)
                        col_weight = col_diff/len(self.df_dict["sens_attrs"])
                        self.weight_dict[col] = col_weight
                        df_new[col] *= col_weight
                        X_val_new[col] *= col_weight
                        outfile.write(col + ": " + str(col_weight) + "\n")
                df_new.to_csv("Datasets/reweigh/" + self.df_dict["filename"] + ".csv", index_label=index)
            elif self.proxy == "remove":
                with open(self.link + "removed_attributes.txt", 'w') as outfile:
                    df_new = copy.deepcopy(self.df)
                    self.weight_dict = dict()
                    cols = list(df_new.columns)
                    cols.remove(self.df_dict["label"])
                    for sens in self.df_dict["sens_attrs"]:
                        cols.remove(sens)

                    for col in cols:
                        cont = False
                        if col in self.allowed:
                            self.weight_dict[col] = 1
                            continue
                        x_arr = df_new[col].to_numpy()
                        col_diff = 0
                        for sens in self.df_dict["sens_attrs"]:
                            z_arr = df_new[sens]
                            pearson = pearsonr(x_arr, z_arr)
                            sens_corr = abs(pearson[0])
                            if math.isnan(sens_corr):
                                sens_corr = 1
                            if sens_corr > 0.5 and pearson[1] < 0.05:
                                X_val_new = X_val_new.loc[:, X_val_new.columns != col]
                                cont = True
                                outfile.write(col + "\n")
                                continue
                        if not cont:
                            self.weight_dict[col] = 1
                    df_new.to_csv("Datasets/removed/" + self.df_dict["filename"] + ".csv", index_label=index)

            X_val_cluster = copy.deepcopy(X_val_new)
            for sens in self.df_dict["sens_attrs"]:
                X_val_cluster = X_val_cluster.loc[:, X_val_cluster.columns != sens]

            #If the clustersize is fixed (hence min and max clustersize has the same value)
            if self.ccr[0] == self.ccr[1] and self.ccr[0] != -1:
                clustersize = self.ccr[0]
            else:
                sens_groups = len(X_val_new.groupby(self.df_dict["sens_attrs"]))
                if self.ccr[0] == -1:
                    min_cluster = min(len(X_val_cluster.columns), int(len(X_val_cluster)/(50*sens_groups)))
                else:
                    min_cluster = self.ccr[0]
                if self.ccr[1] == -1:
                    max_cluster = min(int(len(X_val_cluster.columns)**2/2), int(len(X_val_cluster)/(10*sens_groups)))
                else:
                    max_cluster = self.ccr[1]

                #ELBOW
                #The following implements the Elbow Method, using the KneeLocator to perform the
                #manual step of finding the elbow point.
                if self.cluster_algorithm == "elbow":
                    k_range = range(min_cluster, max_cluster)
                    inertias = []
                    for k in k_range:
                        km = KMeans(n_clusters = k)
                        km.fit(X_val_cluster)
                        inertias.append(km.inertia_)
                    y = np.zeros(len(inertias))

                    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
                    clustersize = kn.knee - 1

                #LOGMEANS
                #Calls the LOGMeans method instead as the parameter estimation algorithm.
                if self.cluster_algorithm == "LOGmeans":
                    clustersize = log_means(X_val_cluster, min_cluster, max_cluster)

            #Save the number of generated clusters as metadata
            with open(self.link + "clustersize.txt", 'w') as outfile:
                outfile.write(str(clustersize))

            #Apply the k-means algorithm on the validation dataset
            if self.proxy in ("no", "remove", "reweigh"):
                self.kmeans = KMeans(clustersize).fit(X_val_cluster)
                cluster_results = self.kmeans.predict(X_val_cluster)
                X_val_cluster["cluster"] = cluster_results
            elif self.proxy == "sfc":
                p, q, k = 1, 3, clustersize
                dataset_path = "Datasets/" + self.input_file + ".csv"
                self.sfc = ScalableClustering(p=p, q=q, k=k, dataset_path=dataset_path)
                self.sfc.dataset = X_val_cluster.to_numpy(dtype=float)
                self.sfc.colors = X_val[self.df_dict["sens_attrs"][0]].to_numpy()
                self.sfc.fit()
                cluster_results = self.sfc.predict(X_val_cluster.to_numpy(dtype=float))
                X_val_cluster["cluster"] = cluster_results
            elif "pfc" in self.proxy:
                if self.proxy == "pfc_greedy":
                    method = "greedy_capture"
                elif self.proxy == "pfc_local":
                    method = "local_search"
                elif self.proxy == "pfc_kmeans":
                    method = "kmeans++"

                self.pfc = PFC(method_type=method, k=clustersize)
                self.pfc.fit(X_val_cluster)
                cluster_results, balance_score, silhouette_avg = self.pfc.predict(X_val_cluster.to_numpy(dtype=float), X_val[self.df_dict["sens_attrs"][0]].to_numpy())
                X_val_cluster["cluster"] = cluster_results

            #Shelve all variables and save it the folder.

            #if clustersize != 1 and self.proxy == "no":
            filename = self.link + "cluster_" + str(self.iteration) + ".out"
            my_shelf = shelve.open(filename, 'n')
            for key in dir():
                try:
                    if "pfc" in self.proxy:
                        my_shelf["kmeans"] = self.pfc
                    elif self.proxy == "sfc":
                        my_shelf["kmeans"] = self.sfc
                    else:
                        my_shelf["kmeans"] = self.kmeans
                except:
                    pass
            my_shelf.close()

            if self.proxy == "no":
                self.weight_dict = None

        else:
            filename = self.link + "cluster_0.out"
            my_shelf = shelve.open(filename)
            self.kmeans = my_shelf["kmeans"]
            my_shelf.close()

            # read the file into a dict
            self.weight_dict = {}

            with open(self.link + "reweighing_attributes.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue  # skip empty or malformed lines
                    key, value = line.split(":", 1)
                    self.weight_dict[key.strip()] = float(value.strip())

            X_val_cluster = copy.deepcopy(X_val)
            for attr in self.df_dict["sens_attrs"]:
                X_val_cluster = X_val_cluster.loc[:, X_val_cluster.columns != attr]

            if self.proxy in ("reweigh", "remove"):
                for col in list(X_val_cluster.columns):
                    if col in self.weight_dict:
                        X_val_cluster[col] *= self.weight_dict[col]
                    else:
                        X_val_cluster = X_val_cluster.loc[:, X_val_cluster.columns != col]

            cluster_results = self.kmeans.predict(X_val_cluster)
            X_val_cluster["cluster"] = cluster_results

        clustered_df = X_val_cluster.groupby("cluster")
        self.model_dict = dict()
        column_list = val_df.columns

        groups = val_df[self.df_dict["sens_attrs"]].drop_duplicates(self.df_dict["sens_attrs"]).reset_index(drop=True)
        actual_num_of_groups = len(groups)
        sensitive_groups = []
        sens_cols = groups.columns
        for i, row in groups.iterrows():
            sens_grp = []
            for col in sens_cols:
                sens_grp.append(row[col])
            if len(sens_grp) == 1:
                sensitive_groups.append(sens_grp[0])
            else:
                sensitive_groups.append(tuple(sens_grp))


        if not self.reg:
            if classes <= 2:
                metricer = Metrics(self.df_dict["sens_attrs"], self.df_dict["label"])
            else:
                metricer = Metrics_Multi(self.df_dict["sens_attrs"], self.df_dict["label"])
        else:
            metricer = Metrics_Reg(self.df_dict["sens_attrs"], self.df_dict["label"])

        for key, item in clustered_df:
            part_df = clustered_df.get_group(key)
            part_df.index.name = "index"
            part_df2 = val_df.merge(part_df, on="index", how="inner")[column_list]
            groups2 = part_df2[self.df_dict["sens_attrs"]].drop_duplicates(self.df_dict["sens_attrs"]).reset_index(drop=True)
            num_of_groups = len(groups2)
            cluster_sensitive_groups = []
            for i, row in groups2.iterrows():
                sens_grp = []
                for col in sens_cols:
                    sens_grp.append(row[col])
                if len(sens_grp) == 1:
                    cluster_sensitive_groups.append(sens_grp[0])
                else:
                    cluster_sensitive_groups.append(tuple(sens_grp))

            #If a cluster does not contain samples of all groups, it will take the k nearest neighbors
            #(default value = 15) to val the model combinations
            if num_of_groups != actual_num_of_groups:
                if "pfc" in self.proxy:
                    cluster_center = self.pfc.model.cluster_centers_[key]
                elif self.proxy == "sfc":
                    cluster_center = self.sfc.kmedoids.cluster_centers_[key]
                else:
                    cluster_center = self.kmeans.cluster_centers_[key]
                grouped_df = X_val.groupby(self.df_dict["sens_attrs"])
                for sens_grp in sensitive_groups:
                    if sens_grp not in cluster_sensitive_groups:
                        knn_df = grouped_df.get_group(sens_grp)
                        for sens_attr in self.df_dict["sens_attrs"]:
                            knn_df = knn_df.loc[:, knn_df.columns != sens_attr]
                        if self.proxy in ("reweigh", "remove"):
                            for col in list(knn_df.columns):
                                if col in self.weight_dict:
                                    knn_df[col] *= self.weight_dict[col]
                                else:
                                    knn_df = knn_df.loc[:, knn_df.columns != col]
                        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(knn_df.values)
                        indices = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=False)
                        real_indices = knn_df.index[indices.flatten()].tolist()
                        nearest_neighbors_df = val_df.loc[real_indices]
                        part_df2 = pd.concat([part_df2, nearest_neighbors_df], ignore_index=True)

            groups = self.df.groupby(self.df_dict["sens_attrs"]).groups.keys()
            if not self.reg:
                comb_list_global, _ = metricer.fairness_metric(part_df2,
                    model_comb, groups, self.df_dict["favored"], self.metric, self.lam, comb_amount=1)
            else:
                comb_list_global, _ = metricer.fairness_metric(part_df2,
                    model_comb, groups, self.df_dict["favored"], self.metric, self.lam, comb_amount=1)

            subdict = dict()
            for i, gt2 in enumerate(list(groups)):
                dict_key = []
                if isinstance(gt2, float) or isinstance(gt2, int):
                    dict_key.append(float(gt2))
                else:
                    gt = tuple(gt2)
                    for j in gt:
                        dict_key.append(float(j))
                subdict[str(dict_key)] = comb_list_global[0][i]

            self.model_dict[key] = subdict

        return self


    #This implementation allows only for the classical training methods (thus not the one including fair methods)
    def predict(self, X_test):
        """This function testicts the label of each testiction sample for FALCC/FALCC-SBT
        (the online phase).

        Parameters
        ----------
        X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
            testiction data vector, where n_samples is the number of samples and
            m_features is the number of features.


        Returns/Output
        ----------
        test_df: Output DataFrame
            Contains: index, value of sensitive attributes, label, predicted value,
            model used for prediction, model combination used for prediction.
        """
        if not self.sbt:
            cluster_model = "FALCC"
        else:
            cluster_model = "FALCC-SBT"

        index = self.df.index.name
        test_df = pd.DataFrame(columns=[index, cluster_model, "model_used"])
        test_count = 0

        sens_count = 1
        X_test_cluster = copy.deepcopy(X_test)
        for attr in self.df_dict["sens_attrs"]:
            test_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1
            X_test_cluster = X_test_cluster.loc[:, X_test_cluster.columns != attr]

        if self.proxy in ("reweigh", "remove"):
            for col in list(X_test_cluster.columns):
                if col in self.weight_dict:
                    X_test_cluster[col] *= self.weight_dict[col]
                else:
                    X_test_cluster = X_test_cluster.loc[:, X_test_cluster.columns != col]

        Z_test = copy.deepcopy(X_test)
        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                Z_test = Z_test.loc[:, Z_test.columns != sens]

        if self.proxy == "sfc":
            cluster_results = self.sfc.predict(X_test_cluster.to_numpy(dtype=float))
        elif "pfc" in self.proxy:
            cluster_results, balance_score, silhouette_avg = self.pfc.predict(X_test_cluster.to_numpy(dtype=float), X_test[self.df_dict["sens_attrs"][0]].to_numpy())

        for i in range(len(X_test)):
            sens_value = []
            for attr in self.df_dict["sens_attrs"]:
                sens_value.append(float(X_test.iloc[i][attr]))

            if self.proxy in ("no", "remove", "reweigh"):
                cluster_results = self.kmeans.predict(X_test_cluster.iloc[i].values.reshape(1, -1))
            
            if self.proxy == "sfc" or "pfc" in self.proxy:
                model = self.model_dict[cluster_results[i]][str(sens_value)]
            else:
                model = self.model_dict[cluster_results[0]][str(sens_value)]
                
            used_model = joblib.load(model)

            if "LinearRegression" in model or "Ridge" in model:
                prediction = used_model.predict(Z_test.iloc[i].values.reshape(1, -1))[0][0]
            else:
                prediction = used_model.predict(Z_test.iloc[i].values.reshape(1, -1))[0]

            test_df.at[test_count, index] = X_test.index[i]
            for attr in self.df_dict["sens_attrs"]:
                test_df.at[test_count, attr] = X_test.iloc[i][attr]
            #test_df.at[test_count, self.df_dict["label"]] = y_test.iloc[i].values[0]
            test_df.at[test_count, cluster_model] = prediction
            test_df.at[test_count, "model_used"] = model

            test_count = test_count + 1

        return list(test_df[cluster_model])
