from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, AdamW
import datetime
import argparse
from hyperopt.pyll.base import scope
from hyperopt import fmin, tpe, hp, Trials, space_eval
from sklearn.model_selection import KFold
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import pickle as pkl
from sklearn.base import BaseEstimator, TransformerMixin

def seed_everything(SEED: int = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def using_gpu(gpu_n):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_n)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5 * 1024)],
            )
        except RuntimeError as e:
            print(e)


def rsquared(y_true, y_pred):
    y_i = np.array(y_true).reshape(-1)
    y_hat = np.array(y_pred).reshape(-1)
    y_mean = np.mean(np.array(y_i))

    SSR = sum((y_i - y_hat)**2)
    SST = sum((y_i - y_mean)**2)
    
    return 1-SSR/SST

# fmt: off
class HyperparameterTuning:
    def __init__(self, model, kfold_n, max_evals, dataset, scaler):#, loss):

        self.SEED = 42
        self.model_name = model
        self.dataset = dataset
        self.max_evals = max_evals
        self.kf = KFold(n_splits=kfold_n, shuffle=True, random_state=self.SEED)
        self.scaler = scaler
        # self.loss = loss

        self.space_mlp = {
            "learning_rate": hp.quniform("learning_rate", 0.0001, 0.01, q=0.0005),
            "rate": hp.quniform("rate", 0.1, 0.5, q=0.1),
            "units": scope.int(hp.quniform("units", 10, 100, q=5)),
            "units1": scope.int(hp.quniform("units1", 10, 100, q=5)),
            "units2": scope.int(hp.quniform("units2", 10, 100, q=5)),
            "units3": scope.int(hp.quniform("units3", 10, 100, q=5)),
            "units4": scope.int(hp.quniform("units4", 10, 100, q=5)),
            "units5": scope.int(hp.quniform("units5", 10, 100, q=5)),
            "units6": scope.int(hp.quniform("units6", 10, 100, q=5)),
            "batch_size": hp.choice("batch_size", [32, 64, 128]),
            "layers": scope.int(hp.quniform("layers", 1, 6, 1)),
            "decay": hp.quniform("decay", 0.0, 0.01, q=0.0001),
            "epsilon": hp.quniform("epsilon", 0.00001, 0.0001, q=0.00001),
            "activ": hp.choice("activ", ["relu", "elu", "swish", tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.gelu, tf.nn.silu]),
            "activ1": hp.choice("activ1", ["relu", "elu", "swish", tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.gelu, tf.nn.silu]),
            "activ2": hp.choice("activ2", ["relu", "elu", "swish", tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.gelu, tf.nn.silu]),
            "activ3": hp.choice("activ3", ["relu", "elu", "swish", tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.gelu, tf.nn.silu]),
            "activ4": hp.choice("activ4", ["relu", "elu", "swish", tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.gelu, tf.nn.silu]),
            "activ5": hp.choice("activ5", ["relu", "elu", "swish", tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.gelu, tf.nn.silu]),
            "activ6": hp.choice("activ6", ["relu", "elu", "swish", tf.nn.crelu, tf.nn.relu6, tf.nn.selu, tf.nn.leaky_relu, tf.nn.tanh, tf.nn.softplus, tf.nn.gelu, tf.nn.silu]),
            "optimizer": hp.choice("optimizer", [Adam, Nadam, RMSprop, AdamW]),
        }

        self.space_rf = {
            "n_estimator": scope.int(hp.quniform("n_estimators", 10, 200, q=5)),
            "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
            "max_depth": scope.int(hp.quniform("max_depth", 5, 20, q=1)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, q=1)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 2, 20, q=1)),
        }

        self.space_svr = {
            "C": hp.quniform("C", 0.1, 10, q=0.1),
            "kernel": hp.choice("kernel", ["sigmoid", "rbf", "linear"]),
            "degree": hp.choice("degree", [2, 3, 4, 5, 6]),
            "epsilon": hp.quniform("epsilon", 0.01, 1.0, q=0.001),
            "gamma": hp.choice("gamma", [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]),
        }

        self.space_xgb = {
            "max_depth": scope.int(hp.quniform("max_depth", 3, 4, q=1)),
            "learning_rate": hp.quniform("learning_rate", 0.001, 0.5, q=0.001),
            "gamma": hp.choice("gamma", [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]),
            "subsample": hp.quniform("subsample", 0.1, 0.5, q=0.05),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.2, 1, q=0.05),
            "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, q=1)),
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 30, q=1)),
        }

    def get_mlp(self, params):
        X_train, *_ = self.dataset
        model = Sequential()
        model.add(Dense(units=params["units"], input_dim=X_train.shape[1], activation=params["activ"]))

        for i in range(1, params["layers"] + 1):
            model.add(Dense(units=params[f"units{i}"], activation=params[f"activ{i}"]))
            model.add(Dropout(rate=params["rate"]))

        model.add(Dense(1, activation="relu"))

        optimizer = params["optimizer"](
            learning_rate=params["learning_rate"], weight_decay=params["decay"], epsilon=params["epsilon"]
        )

        model.compile(optimizer=optimizer, loss="mean_absolute_percentage_error")

        return model

    def get_rf(self, params):
        model = RandomForestRegressor(
            n_estimators=params["n_estimator"],
            max_features=params["max_features"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=self.SEED,
        )

        return model

    def get_svr(self, params):
        model = SVR(
            C=params["C"],
            degree=params["degree"],
            epsilon=params["epsilon"],
            kernel=params["kernel"],
            gamma=params["gamma"],
        )

        return model

    def get_xgb(self, params):
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            gamma=params["gamma"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            learning_rate=params["learning_rate"],
        )

        return model

    def tuning_mlp(self, params):
        val_loss_mean = []
        X_train, _, y_train, _ = self.dataset

        for train_idx, test_idx in self.kf.split(X_train):
            kf_X_train, kf_X_val, kf_y_train, kf_y_val = (
                X_train[train_idx],
                X_train[test_idx],
                y_train[train_idx],
                y_train[test_idx],
            )

            model = self.get_mlp(params)

            es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)
            result = model.fit(
                kf_X_train,
                kf_y_train,
                verbose=0,
                batch_size=params["batch_size"],
                validation_data=(kf_X_val, kf_y_val),
                epochs=3000,
                callbacks=[es],
            )

            validation_loss = np.amin(result.history["val_loss"])
            val_loss_mean.append(validation_loss)

        with open(f"{cwd}/{out_dir}/validation_loss.txt", "a") as f1:
            f1.write(f"{', '.join(map(str, val_loss_mean))} ")
            f1.write(f"Best validation loss: {np.mean(val_loss_mean)}\n")

        print("Best validation loss of epoch:", np.mean(val_loss_mean))
        score = np.mean(val_loss_mean)

        return score

    def tuning_rf(self, params):
        model = self.get_rf(params)
        X_train, X_test, y_train, y_test = self.dataset
        rmse = cross_val_score(model, X_train, y_train, cv=self.kf, scoring="neg_mean_squared_error")

        with open(f"{cwd}/{out_dir}/validation_rmse.txt", "a") as f:
            f.write(f"{', '.join(map(str, rmse))} ")
            f.write(f"Best validation rmse: {rmse.mean()}\n")

        print("Best validation rmse:", rmse.mean())
        score = -rmse.mean()

        return score

    def tuning_svr(self, params):
        model = self.get_svr(params)
        X_train, X_test, y_train, y_test = self.dataset
        rmse = cross_val_score(model, X_train, y_train, cv=self.kf, scoring="neg_mean_squared_error")

        with open(f"{cwd}/{out_dir}/validation_rmse.txt", "a") as f:
            f.write(f"{', '.join(map(str, rmse))} ")
            f.write(f"Best validation rmse: {rmse.mean()}\n")

        print("Best validation rmse:", rmse.mean())
        score = -rmse.mean()

        return score

    def tuning_xgb(self, params):
        model = self.get_xgb(params)
        X_train, X_test, y_train, y_test = self.dataset
        rmse = cross_val_score(model, X_train, y_train, cv=self.kf, scoring="neg_mean_squared_log_error")

        with open(f"{cwd}/{out_dir}/validation_rmse.txt", "a") as f:
            f.write(f"{', '.join(map(str, rmse))} ")
            f.write(f"Best validation rmse: {rmse.mean()}\n")

        print(f"Best validation rmse:", rmse.mean())
        score = -rmse.mean()

        return score

    def run(self):

        if self.model_name == "mlp":
            model = self.tuning_mlp
            space = self.space_mlp

        elif self.model_name in ["rf", "randomforest"]:
            model = self.tuning_rf
            space = self.space_rf

        elif self.model_name == "svr":
            model = self.tuning_svr
            space = self.space_svr

        elif self.model_name in ["xgb", "xgboost"]:
            model = self.tuning_xgb
            space = self.space_xgb

        trials = Trials()
        best = fmin(model, space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials, rstate=np.random.default_rng(self.SEED))
        return space_eval(space, best)

    def result(self, best_params):
        X_train, X_test, y_train, y_test = self.dataset

        if self.model_name == "mlp":
            model = self.get_mlp(best_params)
            es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)
            mc = ModelCheckpoint(
                f"{cwd}/{out_dir}/{out_dir}.h5",
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                mode="min",
            )
            model.fit(
                X_train,
                y_train,
                verbose=0,
                batch_size=best_params["batch_size"],
                validation_data=(X_test, y_test),
                epochs=3000,
                callbacks=[es, mc],
            )

            crelu = tf.nn.crelu
            model = tf.keras.models.load_model(
                f"{cwd}/{out_dir}/{out_dir}.h5", custom_objects={"crelu_v2": crelu}
            )

        else:
            if self.model_name == "rf":
                model = self.get_rf(best_params)
                model.fit(X_train, y_train)
                with open(f"{cwd}/{out_dir}/{out_dir}.pkl", "wb") as f:
                    pkl.dump(model, f)

            elif self.model_name == "svr":
                model = self.get_svr(best_params)
                model.fit(X_train, y_train)
                with open(f"{cwd}/{out_dir}/{out_dir}.pkl", "wb") as f:
                    pkl.dump(model, f)

            elif self.model_name == "xgb":
                model = self.get_xgb(best_params)
                model.fit(X_train, y_train)
                model.save_model(f"{cwd}/{out_dir}/{out_dir}.model")

        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        if self.scaler != "F":
            scaler = pkl.load(open(self.scaler, "rb"))

            y_train = scaler.inverse_transform(y_train)
            y_test = scaler.inverse_transform(y_test)
            y_train_pred = scaler.inverse_transform(y_train_pred)
            y_pred = scaler.inverse_transform(y_pred)

        with open(f"{cwd}/{out_dir}/input_variables.txt", "a") as f:
            f.write("best parameters\n")
            f.write("=====================\n")
            for k, v in best_params.items():
                if k == "optimizer":
                    f.write(f"{k}: {v.__name__}\n")
                else:
                    f.write(f"{k}: {v}\n")

            f.write("\n")
            f.write("statistics\n")
            f.write("=====================\n")
            f.write(f"Train R2: {rsquared(y_train, y_train_pred)}\n")
            f.write(f"Test R2: {rsquared(y_test, y_pred)}\n")
            f.write(f"Train RMSE: {mean_squared_error(y_train, y_train_pred)**0.5}\n")
            f.write(f"Test RMSE: {mean_squared_error(y_test, y_pred)**0.5}\n")
            f.write(f"Train MSE: {mean_squared_error(y_train, y_train_pred)}\n")
            f.write(f"Test MSE: {mean_squared_error(y_test, y_pred)}\n")
            f.write(f"Train MAE: {mean_absolute_error(y_train, y_train_pred)}\n")
            f.write(f"Train MAE: {mean_absolute_error(y_test, y_pred)}")


if __name__ == "__main__":

    SEED = 42
    seed_everything(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--scaler", type=str, default="F")
    # parser.add_argument("--loss", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--bayesian_evals", type=int)
    parser.add_argument("--gpu_n", type=str)
    parser.add_argument("--kfold", type=int)

    args = parser.parse_args()
    model = args.model
    dataset_path = args.dataset_path
    scaler = args.scaler
    # loss = args.loss
    out_dir = args.out_dir
    max_evals = args.bayesian_evals
    gpu_n = args.gpu_n
    kfold = args.kfold


    using_gpu(gpu_n)
    cwd = os.getcwd()

    if out_dir not in os.listdir(cwd):
        os.mkdir(f"{cwd}/{out_dir}")

    with open(f"{cwd}/{out_dir}/input_variables.txt", "w") as f:
        f.write(f"## {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"model: {model}\n")
        f.write(f"dataset: {dataset_path.split('/')[-1]}\n")
        # f.write("last dense: relu\n")
        # f.write(f"loss: {loss}\n")
        f.write(f"max_evals: {max_evals}\n")
        f.write(f"kfold = {kfold}\n\n")

    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)

    print(out_dir)
    tuning = HyperparameterTuning(model=model, kfold_n=kfold, max_evals=max_evals, dataset=dataset, scaler=scaler)#, loss=loss)
    best_params = tuning.run()
    tuning.result(best_params)
