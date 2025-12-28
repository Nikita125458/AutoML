import sys
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Добавляем путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наши модули
from data_preprocessing import load_and_preprocess_data # предобработка данных
from autokeras_experiment_simple import run_autokeras_multiple_seeds # Autokeras (AutoML для нейросетей)
from neural_experiment_simple import SimpleNeuralExperiment # ручные нейросети
from visualize_results import main as visualize_results # визуализация полученных результатов
from data_analysis import main as data_analysis # анализ исходных данных

# подготовка данных
def prepare_data_simple():
    try:
        data_dict = load_and_preprocess_data()
        
        prepared_data = {}
        
        for dataset_name in ['bank', 'housing', 'churn']:
            if dataset_name in data_dict:
                data = data_dict[dataset_name]
                
                # Проверяем, что данные имеют правильный формат
                if not isinstance(data['X_train'], (np.ndarray, pd.DataFrame)):
                    print(f"Предупреждение: X_train для {dataset_name} имеет неправильный тип: {type(data['X_train'])}")
                    continue
                
                # Данные уже предобработаны, нужно только разделить на train/val
                X_train = data['X_train']
                y_train = data['y_train']
                
                # Преобразуем pandas DataFrame в numpy array для AutoKeras
                if hasattr(X_train, 'values'):
                    X_train = X_train.values
                if hasattr(data['X_test'], 'values'):
                    X_test = data['X_test'].values
                else:
                    X_test = data['X_test']
                
                prepared_data[dataset_name] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': data['y_test'],
                    'problem_type': data['problem_type'],
                    'dataset_name': dataset_name
                }
                
                print(f"  {dataset_name}: train={X_train.shape}, test={X_test.shape}")
        
        return prepared_data
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc()
        return None

# Серия A: AutoKeras с разным количеством trials и анализом воспроизводимости (разные seed)
def run_experiment_series_a(data_dict):
    print("Серия A: AutoKeras - зависимость от max_trials (с воспроизводимостью)")
    max_trials_list = [7, 15, 30, 60]
    seeds = [42, 43, 44]
    
    all_results = {}
    for dataset_name, data in data_dict.items():
        print(f"Датасет: {dataset_name}")
        dataset_results = {}
        
        for max_trials in max_trials_list:
            print(f"\nAutoKeras с max_trials={max_trials}")
            
            # Запуск с 3 разными seeds для анализа воспроизводимости
            all_seed_results = run_autokeras_multiple_seeds(X_train=data['X_train'], y_train=data['y_train'], X_test=data['X_test'], y_test=data['y_test'], problem_type=data['problem_type'], dataset_name=dataset_name, max_trials=max_trials, epochs=50, seeds=seeds)
            # Сохраняем все результаты
            dataset_results[max_trials] = {
                'all_seeds': all_seed_results,
                'mean_accuracy': float(np.mean([r['test_metrics']['accuracy'] for r in all_seed_results.values()])),
                'std_accuracy': float(np.std([r['test_metrics']['accuracy'] for r in all_seed_results.values()])),
                'mean_f1': float(np.mean([r['test_metrics']['f1_score'] for r in all_seed_results.values()])),
                'std_f1': float(np.std([r['test_metrics']['f1_score'] for r in all_seed_results.values()]))
            }
            
            # Вывод краткой статистики
            accuracies = [r['test_metrics']['accuracy'] for r in all_seed_results.values()]
            f1_scores = [r['test_metrics']['f1_score'] for r in all_seed_results.values()]
            print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            print(f"  F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")    
            
        all_results[dataset_name] = dataset_results
        
        # Сохраняем промежуточные результаты
        results_dir = "results/series_a"
        os.makedirs(results_dir, exist_ok=True)
        
        dataset_filename = f"{results_dir}/{dataset_name}_results.json"
        with open(dataset_filename, 'w') as f:
            json.dump(dataset_results, f, indent=4)
    
    # Сохраняем все результаты
    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_filename = f"{summary_dir}/series_a_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nВсе результаты Series A сохранены в: {summary_filename}")
    return all_results

# Загружаем результаты AutoKeras из Series A
def load_autokeras_results(dataset_name, max_trials=60, seed=42):
    results_path = f"results/autokeras/{dataset_name}/seed_{seed}/results_{max_trials}trials.json"
    
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return {
            'dataset': dataset_name,
            'problem_type': results.get('problem_type', 'classification'),
            'max_trials': max_trials,
            'seed': seed,
            'training_time_seconds': results.get('training_time_seconds', 0),
            'test_metrics': results.get('test_metrics', {}),
            'best_architecture': results.get('best_architecture', 'unknown'),
            'param_count': results.get('param_count', 0),
            'model_saved': results.get('model_saved', False),
            'model_path': results.get('model_path', None),
            'timestamp': results.get('timestamp', '')
        }
    else:
        print(f"Файл не найден: {results_path}")
        return None

# сравнение AutoKeras (max_trials=60 из Series A) с двумя фиксированными архитектурами
def run_experiment_series_b(data_dict):
    print("Серия B: Сравнение AutoKeras (max_trials=60) и двух ручных архитектур")
    
    all_results = {}
    
    for dataset_name, data in data_dict.items():
        print(f"Датасет: {dataset_name}")
        
        dataset_results = {}
        
        # 1. Загружаем результаты AutoKeras из Series A (max_trials=60)
        print(f"\n1. AutoKeras (max_trials=60 из Series A):")
        ak_results = load_autokeras_results(dataset_name, max_trials=60, seed=42)
        dataset_results['autokeras'] = ak_results                
        if data['problem_type'] == 'classification':
            print(f"   Accuracy: {ak_results['test_metrics']['accuracy']:.4f}, F1: {ak_results['test_metrics']['f1_score']:.4f}")
        else:
            print(f"   R²: {ak_results['test_metrics']['r2_score']:.4f}, RMSE: {ak_results['test_metrics']['rmse']:.4f}")
        print(f"   Время обучения: {ak_results['training_time_seconds']:.0f} сек")
        print(f"   Параметров: {ak_results['param_count']:,}")
        
        # 2. Обучаем две фиксированные архитектуры нейросетей (по 1 часу каждая)
        print(f"\n2. Обучение двух ручных архитектур нейросетей:")
        neural_exp = SimpleNeuralExperiment(X_train=data['X_train'], y_train=data['y_train'], X_test=data['X_test'], y_test=data['y_test'], problem_type=data['problem_type'], dataset_name=dataset_name)
        results_neural, models = neural_exp.run_both_architectures()
        dataset_results.update(results_neural)
        print("\n   Обе ручные архитектуры обучены по 1 часу")
        all_results[dataset_name] = dataset_results
        
        # сохраняем промежуточные результаты
        results_dir = "results/series_b"
        os.makedirs(results_dir, exist_ok=True)
        
        dataset_filename = f"{results_dir}/{dataset_name}_results.json"
        with open(dataset_filename, 'w') as f:
            json.dump(dataset_results, f, indent=4)
            
        print(f"\nРезультаты Series B для {dataset_name} сохранены")
    
    # сохраняем все результаты
    summary_dir = "results/summary"
    os.makedirs(summary_dir, exist_ok=True)

    summary_filename = f"{summary_dir}/series_b_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nВсе результаты Series B сохранены в: {summary_filename}")
    return all_results

# главная функция для запуска всех экспериментов
def main():
    # Создаем необходимые директории
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/autokeras", exist_ok=True)
    os.makedirs("models/optuna", exist_ok=True)
    
    # Загружаем и подготавливаем данные
    data_dict = prepare_data_simple()
    print(f"\nЗагружено {len(data_dict)} датасета")
    
        # Выбор датасетов для экспериментов
    print("\nВыберите датасеты для экспериментов:")
    print("1. Все датасеты")
    print("2. Только bank")
    print("3. Только housing")
    print("4. Только churn")
    
    dataset_choice = input("\nВведите номер датасета (1-5): ").strip()
    
    # Фильтруем датасеты по выбору
    if dataset_choice == '2':
        data_dict = {k: v for k, v in data_dict.items() if k == 'bank'}
    elif dataset_choice == '3':
        data_dict = {k: v for k, v in data_dict.items() if k == 'housing'}
    elif dataset_choice == '4':
        data_dict = {k: v for k, v in data_dict.items() if k == 'churn'}
    elif dataset_choice != '1':
        print("Неверный выбор датасета! Используются все датасеты.")
    
    print(f"\nБудут использованы датасеты: {list(data_dict.keys())}")

    # Выбор экспериментов для запуска
    print("\nВыберите эксперименты для запуска:")
    print("1. Series A: AutoKeras с разным количеством trials (с воспроизводимостью)")
    print("2. Series B: Сравнение AutoKeras с двумя ручными архитектурами")
    print("3. Все эксперименты")
    print("4. Только визуализация результатов")
    print("5. Анализ исходных данных")
    
    choice = input("\nВведите номер (1-4): ").strip()
    
    # Только Series A
    if choice == '1':
        run_experiment_series_a(data_dict)
    
    # Только Series B
    elif choice == '2':
        run_experiment_series_b(data_dict)
    
    # Все эксперименты
    elif choice == '3':
        print("Запуск всех экспериментов")
        # Запускаем по порядку
        run_experiment_series_a(data_dict)
        run_experiment_series_b(data_dict)
    
    # Только визуализация
    elif choice == '4':
        visualize_results()
    
    elif choice == '5':
        print("Анализ исходных данных")
        data_analysis()
    
    else:
        print("Неверный выбор!")
        return
    
    print("Эксперименты завершены")

if __name__ == "__main__":
    main()