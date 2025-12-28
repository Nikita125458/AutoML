import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Загрузка и предобработка данных
def load_and_preprocess_data():
    # Загрузка данных
    df_bank = pd.read_csv("data/raw/Bank Marketing.csv")
    df_housing = pd.read_csv("data/raw/housing.csv")
    df_churn = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Определение целевых переменных
    target_col_bank = 'Subscription'
    target_col_housing = 'median_house_value'
    target_col_churn = 'Churn'
    
    # Bank Marketing Dataset
    X_bank = df_bank.drop(columns=[target_col_bank])
    y_bank = df_bank[target_col_bank]
    
    # Housing Dataset
    X_housing = df_housing.drop(columns=[target_col_housing])
    y_housing = df_housing[target_col_housing]

    # Churn Dataset
    X_churn = df_churn.drop(columns=[target_col_churn])
    y_churn = df_churn[target_col_churn]

    # 1. Bank Marketing Dataset
    # Разделение на train/test
    X_bank_train, X_bank_test, y_bank_train, y_bank_test = train_test_split(X_bank, y_bank, test_size=0.2, random_state=42, stratify=y_bank)
    
    # Сохраняем названия колонок
    bank_columns = X_bank_train.columns.tolist()
    
    # Затем предобработка
    categorical_bank = ['Job', 'Marital Status', 'Education', 'Credit', 'Housing Loan', 'Personal Loan', 'Contact', 'Last Contact Month', 'Poutcome']
    
    # Кодирование категориальных признаков
    for col in categorical_bank:
        if col in X_bank_train.columns:
            # Объединяем train и test для кодирования, чтобы получить все категории
            combined = pd.concat([X_bank_train[col], X_bank_test[col]], axis=0)
            combined = combined.astype(str).fillna('unknown')
            le = LabelEncoder()
            le.fit(combined)
            
            # Применяем к train и test
            X_bank_train[col] = le.transform(X_bank_train[col].astype(str).fillna('unknown'))
            X_bank_test[col] = le.transform(X_bank_test[col].astype(str).fillna('unknown'))
    
    # Целевая переменная
    y_bank_train = y_bank_train.replace({2: 0, 1: 1})
    y_bank_test = y_bank_test.replace({2: 0, 1: 1})
    
    # 2. Housing Dataset
    # Разделение на train/test
    X_housing_train, X_housing_test, y_housing_train, y_housing_test = train_test_split(
        X_housing, y_housing, test_size=0.2, random_state=42
    )
    
    # Сохраняем названия колонок
    housing_columns = X_housing_train.columns.tolist()

    # Обработка пропущенных значений
    if 'total_bedrooms' in X_housing_train.columns:
        imputer = SimpleImputer(strategy='median')
        X_housing_train['total_bedrooms'] = imputer.fit_transform(
            X_housing_train[['total_bedrooms']]
        ).ravel()
        X_housing_test['total_bedrooms'] = imputer.transform(
            X_housing_test[['total_bedrooms']]
        ).ravel()
    
    # Кодирование ocean_proximity
    if 'ocean_proximity' in X_housing_train.columns:
        # Объединяем train и test для кодирования
        combined = pd.concat([X_housing_train['ocean_proximity'], 
                             X_housing_test['ocean_proximity']], axis=0)
        le_housing = LabelEncoder()
        le_housing.fit(combined)
        
        X_housing_train['ocean_proximity'] = le_housing.transform(X_housing_train['ocean_proximity'])
        X_housing_test['ocean_proximity'] = le_housing.transform(X_housing_test['ocean_proximity'])
    
    # сиандартизация для housing
    scaler_X_housing = StandardScaler()
    X_housing_train_scaled = scaler_X_housing.fit_transform(X_housing_train)
    X_housing_test_scaled = scaler_X_housing.transform(X_housing_test)
    
    # Преобразование массивов numpy обратно в DataFrame с сохранением названий колонок
    X_housing_train_scaled = pd.DataFrame(X_housing_train_scaled, columns=housing_columns)
    X_housing_test_scaled = pd.DataFrame(X_housing_test_scaled, columns=housing_columns)
    
    # Стандартизация целевой переменной для регрессии
    scaler_y_housing = StandardScaler()
    y_housing_train_scaled = scaler_y_housing.fit_transform(y_housing_train.values.reshape(-1, 1)).ravel()
    y_housing_test_scaled = scaler_y_housing.transform(y_housing_test.values.reshape(-1, 1)).ravel()
    
    # Преобразование целевых переменных в Series
    y_housing_train_scaled = pd.Series(y_housing_train_scaled, name='median_house_value_scaled')
    y_housing_test_scaled = pd.Series(y_housing_test_scaled, name='median_house_value_scaled')
    
    # 3. Churn Dataset
    # Разделение на train/test
    X_churn_train, X_churn_test, y_churn_train, y_churn_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn)
    
    # Сохраняем названия колонок
    churn_columns = [col for col in X_churn_train.columns if col != 'customerID']
    
    # Предобработка
    # Удаление customerID
    if 'customerID' in X_churn_train.columns:
        X_churn_train = X_churn_train.drop(columns=['customerID'])
        X_churn_test = X_churn_test.drop(columns=['customerID'])
    
    # Преобразование TotalCharges
    if 'TotalCharges' in X_churn_train.columns:
        # Преобразуем ' ' в NaN
        X_churn_train['TotalCharges'] = X_churn_train['TotalCharges'].replace(' ', np.nan)
        X_churn_test['TotalCharges'] = X_churn_test['TotalCharges'].replace(' ', np.nan)
        
        X_churn_train['TotalCharges'] = pd.to_numeric(X_churn_train['TotalCharges'])
        X_churn_test['TotalCharges'] = pd.to_numeric(X_churn_test['TotalCharges'])
        
        # Заполняем пропуски медианой (вычисленной по train)
        median_value = X_churn_train['TotalCharges'].median()
        X_churn_train['TotalCharges'] = X_churn_train['TotalCharges'].fillna(median_value)
        X_churn_test['TotalCharges'] = X_churn_test['TotalCharges'].fillna(median_value)
    
    # Кодирование категориальных признаков
    categorical_churn = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_churn:
        if col in X_churn_train.columns:
            # Объединяем train и test для кодирования
            combined = pd.concat([X_churn_train[col], X_churn_test[col]], axis=0)
            combined = combined.astype(str)
            
            # Заполняем пропуски модой (вычисленной по train)
            mode_value = X_churn_train[col].astype(str).mode()[0]
            X_churn_train[col] = X_churn_train[col].astype(str).fillna(mode_value)
            X_churn_test[col] = X_churn_test[col].astype(str).fillna(mode_value)
            
            le = LabelEncoder()
            le.fit(pd.concat([X_churn_train[col], X_churn_test[col]], axis=0))
            
            X_churn_train[col] = le.transform(X_churn_train[col])
            X_churn_test[col] = le.transform(X_churn_test[col])
    
    # Кодирование целевой переменной
    le_y_churn = LabelEncoder()
    y_churn_train = le_y_churn.fit_transform(y_churn_train)  # 'No' -> 0, 'Yes' -> 1
    y_churn_test = le_y_churn.transform(y_churn_test)
    
    # Преобразование целевых переменных в Series
    y_bank_train = pd.Series(y_bank_train, name='Subscription')
    y_bank_test = pd.Series(y_bank_test, name='Subscription')
    
    y_churn_train = pd.Series(y_churn_train, name='Churn')
    y_churn_test = pd.Series(y_churn_test, name='Churn')
    
    # Возвращаем все четыре датасета
    return {
        'bank': {
            'X_train': X_bank_train,
            'X_test': X_bank_test,
            'y_train': y_bank_train,
            'y_test': y_bank_test,
            'problem_type': 'classification',
            'dataset_name': 'bank',
            'scaler_X': None,
            'scaler_y': None
        },
        'housing': {
            'X_train': X_housing_train_scaled,
            'X_test': X_housing_test_scaled,
            'y_train': y_housing_train_scaled,
            'y_test': y_housing_test_scaled,
            'problem_type': 'regression',
            'dataset_name': 'housing',
            'scaler_X': scaler_X_housing,
            'scaler_y': scaler_y_housing,
            'original_y_train': y_housing_train,
            'original_y_test': y_housing_test,
            'feature_names': housing_columns
        },
        'churn': {
            'X_train': X_churn_train,
            'X_test': X_churn_test,
            'y_train': y_churn_train,
            'y_test': y_churn_test,
            'problem_type': 'classification',
            'dataset_name': 'churn',
            'scaler_X': None,
            'scaler_y': None
        }
    }