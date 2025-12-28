import autokeras as ak
import tensorflow as tf
import numpy as np
import random
import time
import os
import json
import tempfile
import shutil
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score

if not hasattr(np, 'object'):
    np.object = object

# Запуск AutoKeras с фиксированным seed для воспроизводимости
def run_autokeras_with_seed(X_train, y_train, X_test, y_test, problem_type, dataset_name, max_trials=10, epochs=50, seed=42):
    print(f"AutoKeras для {dataset_name}, max_trials={max_trials}, seed={seed}")
    
    # фиксация seed для воспроизводимости
    print(f"Установка random seed: {seed}")
    
    # python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    start_time = time.time()
    
    # Создаем временную директорию для AutoKeras проекта
    temp_dir = tempfile.mkdtemp()
    project_name = f"ak_{dataset_name}_{max_trials}trials_seed{seed}"
    
    # Преобразуем данные в numpy массивы с правильными типами
    X_train = X_train.to_numpy() if hasattr(X_train, 'to_numpy') else X_train
    X_test = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test
    y_train = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train
    y_test = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test
    
    # Исправляем типы данных для AutoKeras
    if isinstance(X_train, np.ndarray) and (X_train.dtype == np.object_ or X_train.dtype == object):
        X_train = X_train.astype(np.float32)
    if isinstance(X_test, np.ndarray) and (X_test.dtype == np.object_ or X_test.dtype == object):
        X_test = X_test.astype(np.float32)
    
    # Для классификации: y должен быть int
    if problem_type == 'classification':
        if isinstance(y_train, np.ndarray) and (y_train.dtype == np.object_ or y_train.dtype == object):
            y_train = y_train.astype(np.int32)
        if isinstance(y_test, np.ndarray) and (y_test.dtype == np.object_ or y_test.dtype == object):
            y_test = y_test.astype(np.int32)
    else:  # regression
        if isinstance(y_train, np.ndarray) and (y_train.dtype == np.object_ or y_train.dtype == object):
            y_train = y_train.astype(np.float32)
        if isinstance(y_test, np.ndarray) and (y_test.dtype == np.object_ or y_test.dtype == object):
            y_test = y_test.astype(np.float32)
    
    if problem_type == 'classification':
        # Определяем бинарная или мультиклассовая классификация
        num_classes = len(np.unique(y_train))
        
        if num_classes == 2:
            model = ak.StructuredDataClassifier(max_trials=max_trials, tuner='random', overwrite=True, seed=seed, project_name=project_name, directory=temp_dir, metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split=0.2, validation_data=None)
        else:
            model = ak.StructuredDataClassifier(max_trials=max_trials, tuner='random', overwrite=True, seed=seed, project_name=project_name, directory=temp_dir, metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split=0.2, validation_data=None)
        
        y_pred = model.predict(X_test, verbose=0)
        y_pred_proba = model.predict(X_test, verbose=0)
        
        # Метрики для классификации
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        test_metrics = {'accuracy': float(accuracy), 'f1_score': float(f1), 'num_classes': int(num_classes)}
        
        # Для бинарной классификации добавляем ROC-AUC
        if num_classes == 2 and y_pred_proba.shape[1] == 2:
            try:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                test_metrics['roc_auc'] = float(auc)
            except:
                test_metrics['roc_auc'] = None       
    else:
        model = ak.StructuredDataRegressor(max_trials=max_trials, overwrite=True, seed=seed, project_name=project_name, directory=temp_dir, metrics=['mse'])
        model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split=0.2, validation_data=None)
        y_pred = model.predict(X_test, verbose=0)
        
        # Метрики для регрессии
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        test_metrics = {'mse': float(mse), 'rmse': float(rmse), 'r2_score': float(r2)}
    training_time = time.time() - start_time
    
    # Получаем информацию о лучшей архитектуре
    best_model = model.export_model()
    param_count = best_model.count_params()
    
    # Формируем строковое описание архитектуры
    arch_lines = []
    for layer in best_model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Dense':
            config = layer.get_config()
            units = config.get('units', '?')
            activation = config.get('activation', '?')
            arch_lines.append(f"{layer_type}(units={units}, activation={activation})")
        elif layer_type not in ['InputLayer']:
            arch_lines.append(f"{layer_type}()")
    
    best_architecture = " → ".join(arch_lines) if arch_lines else "unknown"
    
    # Сохраняем модель в отдельную директорию для seed
    models_dir = f"models/autokeras/{dataset_name}/seed_{seed}"
    results_dir = f"results/autokeras/{dataset_name}/seed_{seed}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Сохраняем модель
    model_filename = f"{models_dir}/model_{max_trials}trials.keras"
    best_model.save(model_filename)
    
    # Конвертируем numpy типы для JSON
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Подготавливаем результаты
    results = {
        'dataset': dataset_name,
        'problem_type': problem_type,
        'max_trials': max_trials,
        'epochs': epochs,
        'seed': seed,
        'training_time_seconds': float(training_time),
        'test_metrics': convert_types(test_metrics),
        'best_architecture': best_architecture,
        'param_count': int(param_count),
        'model_path': model_filename,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Сохраняем результаты
    results_filename = f"{results_dir}/results_{max_trials}trials.json"
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Выводим краткую сводку
    print(f"\nВремя обучения: {training_time:.2f} сек")
    if problem_type == 'classification':
        print(f"Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
    else:
        print(f"R²: {test_metrics['r2_score']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
    print(f"Параметров модели: {param_count:,}")
    print(f"Архитектура: {best_architecture}")
    print(f"Результаты сохранены в: {results_filename}")
    # Очищаем временную директорию
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    return results

# Запуск AutoKeras с несколькими seed для анализа воспроизводимости
def run_autokeras_multiple_seeds(X_train, y_train, X_test, y_test, problem_type, dataset_name, max_trials=10, epochs=50, seeds=[42, 43, 44]):
    print(f"Анализ воспроизводимости для {dataset_name}, max_trials={max_trials}")
    print(f"Seeds: {seeds}")
    
    all_results = {}
    for seed in seeds:
        print(f"\nЗапуск с seed={seed}")
        results = run_autokeras_with_seed(X_train, y_train, X_test, y_test, problem_type, dataset_name, max_trials=max_trials,epochs=epochs, seed=seed)
        all_results[seed] = results
    
    # Анализ результатов разных seeds
    if len(all_results) > 1:
        print(f"Анализ воспроизводимости ({len(all_results)} seeds)")
        
        # Собираем метрики
        accuracies = []
        f1_scores = []
        times = []
        for seed, results in all_results.items():
            accuracies.append(results['test_metrics']['accuracy'])
            f1_scores.append(results['test_metrics']['f1_score'])
            times.append(results['training_time_seconds'])
        
        # Вывод статистики
        print(f"Accuracy: среднее={np.mean(accuracies):.4f}, std={np.std(accuracies):.4f}, min={np.min(accuracies):.4f}, max={np.max(accuracies):.4f}")
        print(f"F1 Score: среднее={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}, min={np.min(f1_scores):.4f}, max={np.max(f1_scores):.4f}")
        print(f"Время: среднее={np.mean(times):.2f}с, std={np.std(times):.2f}с")
        
        # Сохраняем сводные результаты
        summary_dir = f"results/autokeras/{dataset_name}/summary"
        os.makedirs(summary_dir, exist_ok=True)
        
        summary = {
            'dataset': dataset_name,
            'max_trials': max_trials,
            'seeds': seeds,
            'seed_results': all_results,
            'statistics': {
                'accuracy_mean': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'accuracy_min': float(np.min(accuracies)),
                'accuracy_max': float(np.max(accuracies)),
                'f1_mean': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores)),
                'f1_min': float(np.min(f1_scores)),
                'f1_max': float(np.max(f1_scores)),
                'time_mean': float(np.mean(times)),
                'time_std': float(np.std(times))
            }
        }
        
        summary_filename = f"{summary_dir}/summary_{max_trials}trials.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"Сводные результаты сохранены в: {summary_filename}")
    
    return all_results