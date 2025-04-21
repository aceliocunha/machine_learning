from LinearReg import LinearReg
from LinearRegGd import LinearRegGd
from utils import normaliza
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score

def compute_regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

def compute_metrics(y_true, y_pred):
    
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred,zero_division=0)
    prec =  precision_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec
def train_test_RegGd(X,y, Model=LinearRegGd):
    
    kf = KFold(n_splits=5)
    param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1], 'epochs': [100, 500, 1000]}
    
    best_r2 = -np.inf
    best_params = {}
    r2_total = []
    mae_total = []
    rmse_total = []
    tempo_total = []
    
    for v in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=v)
        X_train, X_test = normaliza(X_train, X_test)
    
        for lr in param_grid['learning_rate']:
            for epochs in param_grid['epochs']:
                avg_r2 = 0.0
    
                for train_index, val_index in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
                    model = LinearRegGd(lr=lr, epochs=epochs)
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    fold_r2 = r2_score(y_val_fold, y_pred)
                    avg_r2 += fold_r2
    
                avg_r2 /= kf.get_n_splits()
                if avg_r2 > best_r2:
                    best_r2 = avg_r2
                    best_params = {'learning_rate': lr, 'epochs': epochs}
    
        # Treino final com melhores parâmetros encontrados
        start_time = time.time()
        model = LinearRegGd(lr=best_params['learning_rate'], epochs=best_params['epochs'])
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        r2, mae, rmse = compute_regression_metrics(y_test, y_pred_test)
        end_time = time.time()
    
        r2_total.append(r2)
        mae_total.append(mae)
        rmse_total.append(rmse)
        tempo_total.append(end_time - start_time)
    
    print("Melhores parâmetros:", best_params)
    print("Tempo médio de execução:", np.mean(tempo_total))
    print("Desvio padrão do tempo:", np.std(tempo_total))
    print("Tempo total:", sum(tempo_total))
    
    df_results = pd.DataFrame({
        "R²": r2_total,
        "MAE": mae_total,
        "RMSE": rmse_total,
        "Tempo": tempo_total
    })
    
    return print(df_results.aggregate({'R²': ['mean', 'std'], 'MAE': ['mean', 'std'], 'RMSE': ['mean', 'std'], 'Tempo':['mean', 'std']}))
	


def train_tesRg_yint(X,y, Model=LinearReg):
    
    kf = KFold(n_splits=5)
    #param_grid = {'epochs': [1]}
    
    best_params = {}
    r2_total = []
    mae_total = []
    rmse_total = []
    tempo_total = []
    
    for v in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=v)
        X_train, X_test = normaliza(X_train, X_test)
    

        start_time = time.time()
        model = Model()
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        r2, mae, rmse = compute_metrics(y_test, y_pred_test)
        end_time = time.time()
    
        r2_total.append(r2)
        mae_total.append(mae)
        rmse_total.append(rmse)
        tempo_total.append(end_time - start_time)
    

    print("Tempo médio de execução:", np.mean(tempo_total))
    print("Desvio padrão do tempo:", np.std(tempo_total))
    print("Tempo total:", sum(tempo_total))
    
    df_results = pd.DataFrame({
        "R²": r2_total,
        "MAE": mae_total,
        "RMSE": rmse_total,
        "Tempo": tempo_total
    })
    print('############################# Resultados ########################################')
    return print(df_results.aggregate({'R²': ['mean', 'std'], 'MAE': ['mean', 'std'], 'RMSE': ['mean', 'std'], 'Tempo': ['mean', 'std']}))
def train_tesRg_yflot(X,y, Model=LinearReg):
    
    kf = KFold(n_splits=5)
    #param_grid = {'epochs': [1]}
    
    best_params = {}
    r2_total = []
    mae_total = []
    rmse_total = []
    tempo_total = []
    
    for v in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=v)
        X_train, X_test = normaliza(X_train, X_test)
    
        # Treino final com melhores parâmetros encontrados
        start_time = time.time()
        model = Model()
        model.fit(X_train, y_train)
        y_pred_test = model.predict_real(X_test)
        r2, mae, rmse = compute_regression_metrics(y_test, y_pred_test)
        end_time = time.time()
    
        r2_total.append(r2)
        mae_total.append(mae)
        rmse_total.append(rmse)
        tempo_total.append(end_time - start_time)
    
    print("Tempo médio de execução:", np.mean(tempo_total))
    print("Desvio padrão do tempo:", np.std(tempo_total))
    print("Tempo total:", sum(tempo_total))
    
    df_results = pd.DataFrame({
        "R²": r2_total,
        "MAE": mae_total,
        "RMSE": rmse_total,
        "Tempo": tempo_total
    })
    print('############################# Resultados ########################################')
    return print(df_results.aggregate({'R²': ['mean', 'std'], 'MAE': ['mean', 'std'], 'RMSE': ['mean', 'std'], 'Tempo': ['mean', 'std']}))