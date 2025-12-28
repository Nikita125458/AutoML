import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

TENSORFLOW_AVAILABLE = True

def load_autokeras_results():
    """Загружает все результаты AutoKeras из файловой структуры"""
    print("📊 Загрузка результатов AutoKeras из папки results/autokeras/")
    
    base_path = Path("results/autokeras")
    results = {}
    
    if not base_path.exists():
        print("⚠ Папка results/autokeras/ не найдена")
        print("   Сначала запустите эксперименты Series A")
        return {}
    
    # Сканируем датасеты (bank, housing, churn)
    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            results[dataset_name] = {}
            
            print(f"\n  Датасет: {dataset_name}")
            
            # Сканируем папки с seeds (seed_42, seed_43, seed_44)
            seed_dirs = list(dataset_dir.glob('seed_*'))
            if not seed_dirs:
                print(f"    ⚠ Нет папок с seeds")
                continue
            
            # Для каждого trials значения (7, 15, 30, 60)
            trial_values = []
            
            for seed_dir in seed_dirs:
                if seed_dir.is_dir():
                    seed = seed_dir.name.replace('seed_', '')
                    
                    # Находим все файлы results
                    result_files = list(seed_dir.glob('results_*trials.json'))
                    
                    for result_file in result_files:
                        # Извлекаем количество trials
                        filename = result_file.stem
                        trials = int(filename.replace('results_', '').replace('trials', ''))
                        
                        if trials not in trial_values:
                            trial_values.append(trials)
                        
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # Инициализируем структуру если нужно
                            if trials not in results[dataset_name]:
                                results[dataset_name][trials] = {}
                            
                            results[dataset_name][trials][seed] = data
                            
                        except Exception as e:
                            print(f"    ⚠ Ошибка загрузки {result_file.name}: {e}")
            
            trial_values.sort()
            print(f"    Trials: {trial_values}")
            print(f"    Seeds: {[d.name.replace('seed_', '') for d in seed_dirs]}")
    
    print(f"\n✓ Загружено {len(results)} датасетов")
    return results

def load_neural_results():
    """Загружает все результаты ручных нейросетей из файловой структуры"""
    print("\n📊 Загрузка результатов ручных нейросетей из папки results/neural/")
    
    base_path = Path("results/neural")
    results = {}
    
    if not base_path.exists():
        print("⚠ Папка results/neural/ не найдена")
        print("   Сначала запустите эксперименты Series B")
        return {}
    
    # Сканируем датасеты
    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            results[dataset_name] = {}
            
            print(f"\n  Датасет: {dataset_name}")
            
            # Ищем файлы архитектур
            arch_files = list(dataset_dir.glob('*_results.json'))
            
            for arch_file in arch_files:
                arch_name = arch_file.stem.replace('_results', '')
                
                try:
                    with open(arch_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    results[dataset_name][arch_name] = data
                    print(f"    ✓ Архитектура: {arch_name}")
                    
                except Exception as e:
                    print(f"    ⚠ Ошибка загрузки {arch_file.name}: {e}")
    
    print(f"\n✓ Загружено {len(results)} датасетов с ручными архитектурами")
    return results

def print_architectures():
    """Выводит архитектуры нейросетей из файлов .keras"""
    print("\n" + "="*80)
    print("🏗️  АРХИТЕКТУРЫ НЕЙРОСЕТЕЙ")
    print("="*80)
    
    # Проверяем наличие TensorFlow
    if not TENSORFLOW_AVAILABLE:
        print("⚠ TensorFlow не установлен. Не могу загрузить модели .keras")
        print("   Установите TensorFlow: pip install tensorflow")
        return
    
    # Словарь для перевода названий датасетов
    dataset_names = {
        'bank': 'Bank Marketing',
        'housing': 'California Housing',
        'churn': 'Telecom Churn'
    }
    
    # 1. Архитектуры AutoKeras (только seed_42, trials=60)
    print("\n1. АРХИТЕКТУРЫ AUTOKERAS (seed_42, 60 trials):")
    print("-"*60)
    
    ak_base_path = Path("models/autokeras")
    if not ak_base_path.exists():
        print("⚠ Папка models/autokeras/ не найдена")
    else:
        # Для каждого датасета
        for dataset in ['bank', 'housing', 'churn']:
            model_path = ak_base_path / dataset / 'seed_42' / 'model_60trials.keras'
            
            if model_path.exists():
                try:
                    print(f"\n📊 Датасет: {dataset_names.get(dataset, dataset.upper())}")
                    print(f"📁 Модель: {model_path}")
                    
                    # Загружаем модель
                    model = tf.keras.models.load_model(model_path)
                    
                    # Выводим сводку архитектуры
                    print("\n" + "-"*40)
                    print("Сводка архитектуры:")
                    print("-"*40)
                    
                    # Собираем информацию о слоях
                    total_params = 0
                    trainable_params = 0
                    non_trainable_params = 0
                    
                    print(f"{'Слой':<25} {'Выходной размер':<20} {'Параметры':<10}")
                    print("-"*60)
                    
                    for i, layer in enumerate(model.layers):
                        output_shape = str(layer.output_shape)
                        if len(output_shape) > 30:
                            output_shape = output_shape[:27] + "..."
                        
                        params = layer.count_params()
                        total_params += params
                        
                        # Определяем тип параметров
                        if layer.trainable:
                            trainable_params += params
                        else:
                            non_trainable_params += params
                        
                        print(f"{layer.name:<25} {output_shape:<20} {params:<10,}")
                    
                    print("-"*60)
                    print(f"Всего параметров: {total_params:,}")
                    print(f"Обучаемые параметры: {trainable_params:,}")
                    print(f"Необучаемые параметры: {non_trainable_params:,}")
                    
                    # Информация о входе и выходе
                    print(f"\nВходной размер: {model.input_shape}")
                    print(f"Выходной размер: {model.output_shape}")
                    
                except Exception as e:
                    print(f"⚠ Ошибка загрузки модели {dataset}: {e}")
            else:
                print(f"⚠ Модель для датасета {dataset} не найдена")
    
    # 2. Ручные архитектуры (architecture1 и architecture2)
    print("\n\n2. РУЧНЫЕ АРХИТЕКТУРЫ:")
    print("-"*60)
    
    neural_base_path = Path("models/neural")
    if not neural_base_path.exists():
        print("⚠ Папка models/neural/ не найдена")
    else:
        # Для каждого датасета
        for dataset in ['bank', 'housing', 'churn']:
            dataset_path = neural_base_path / dataset
            
            if dataset_path.exists():
                print(f"\n📊 Датасет: {dataset_names.get(dataset, dataset.upper())}")
                print("="*50)
                
                # Проверяем обе архитектуры
                for arch_num in [1, 2]:
                    model_path = dataset_path / f'architecture{arch_num}.keras'
                    
                    if model_path.exists():
                        try:
                            print(f"\n🏗️  Архитектура {arch_num}:")
                            print(f"📁 Модель: {model_path}")
                            
                            # Загружаем модель
                            model = tf.keras.models.load_model(model_path)
                            
                            # Выводим сводку архитектуры
                            print("\n" + "-"*40)
                            print("Сводка архитектуры:")
                            print("-"*40)
                            
                            # Собираем информацию о слоях
                            total_params = 0
                            trainable_params = 0
                            non_trainable_params = 0
                            
                            print(f"{'Слой':<25} {'Тип':<20} {'Выходной размер':<25} {'Параметры':<10}")
                            print("-"*80)
                            
                            for i, layer in enumerate(model.layers):
                                layer_type = layer.__class__.__name__
                                output_shape = str(layer.output_shape)
                                if len(output_shape) > 25:
                                    output_shape = output_shape[:22] + "..."
                                
                                params = layer.count_params()
                                total_params += params
                                
                                # Определяем тип параметров
                                if layer.trainable:
                                    trainable_params += params
                                else:
                                    non_trainable_params += params
                                
                                print(f"{layer.name:<25} {layer_type:<20} {output_shape:<25} {params:<10,}")
                            
                            print("-"*80)
                            print(f"Всего параметров: {total_params:,}")
                            print(f"Обучаемые параметры: {trainable_params:,}")
                            print(f"Необучаемые параметры: {non_trainable_params:,}")
                            
                            # Дополнительная информация
                            print(f"\nВходной размер: {model.input_shape}")
                            print(f"Выходной размер: {model.output_shape}")
                            
                            # Информация об активации выходного слоя
                            if len(model.layers) > 0:
                                last_layer = model.layers[-1]
                                if hasattr(last_layer, 'activation'):
                                    print(f"Активация выходного слоя: {last_layer.activation.__name__}")
                            
                        except Exception as e:
                            print(f"⚠ Ошибка загрузки модели architecture{arch_num}: {e}")
                    else:
                        print(f"⚠ Архитектура {arch_num} не найдена для датасета {dataset}")
                
                print()

def get_problem_type(dataset_name, ak_results):
    """Определяет тип задачи для датасета"""
    # Проверяем первый найденный результат
    for trials_data in ak_results.get(dataset_name, {}).values():
        for seed_data in trials_data.values():
            if 'problem_type' in seed_data:
                return seed_data['problem_type']
    
    # Если не нашли в AutoKeras, проверяем в нейросетях
    neural_path = Path(f"results/neural/{dataset_name}")
    if neural_path.exists():
        for arch_file in neural_path.glob('*_results.json'):
            try:
                with open(arch_file, 'r') as f:
                    data = json.load(f)
                    if 'problem_type' in data:
                        return data['problem_type']
            except:
                pass
    
    # Определяем по имени датасета
    if dataset_name == 'housing':
        return 'regression'
    else:
        return 'classification'

def get_main_metric(data, problem_type):
    """Получает основную метрику в зависимости от типа задачи"""
    if 'test_metrics' not in data:
        return 0
    
    test_metrics = data['test_metrics']
    
    if problem_type == 'classification':
        return test_metrics.get('accuracy', 0)
    else:  # regression
        return test_metrics.get('r2_score', 0)

def get_secondary_metric(data, problem_type):
    """Получает второстепенную метрику в зависимости от типа задачи"""
    if 'test_metrics' not in data:
        return 0
    
    test_metrics = data['test_metrics']
    
    if problem_type == 'classification':
        return test_metrics.get('f1_score', 0)
    else:  # regression
        return test_metrics.get('rmse', 0)

def plot_autokeras_results(ak_results):
    """Строит графики для AutoKeras"""
    if not ak_results:
        print("\n❌ Нет данных AutoKeras для визуализации")
        return
    
    print("\n" + "="*80)
    print("ГРАФИКИ ДЛЯ AUTOKERAS")
    print("="*80)
    
    os.makedirs("results/plots", exist_ok=True)
    
    for dataset_name, trials_data in ak_results.items():
        if not trials_data:
            continue
        
        print(f"\n  📊 Датасет: {dataset_name.upper()}")
        
        # Определяем тип задачи
        problem_type = get_problem_type(dataset_name, ak_results)
        main_metric_name = "R² Score" if problem_type == 'regression' else "Accuracy"
        secondary_metric_name = "RMSE" if problem_type == 'regression' else "F1 Score"
        
        # Собираем данные
        trials_list = sorted(trials_data.keys())
        
        if len(trials_list) < 2:
            print(f"    ⚠ Недостаточно разных trials для графика")
            continue
        
        # Для каждого trials вычисляем среднее по seeds
        main_metrics = []
        secondary_metrics = []
        times = []
        
        for trials in trials_list:
            seeds_data = trials_data[trials]
            
            if not seeds_data:
                continue
            
            # Вычисляем средние значения по seeds
            main_values = []
            secondary_values = []
            time_values = []
            
            for seed_data in seeds_data.values():
                main_values.append(get_main_metric(seed_data, problem_type))
                secondary_values.append(get_secondary_metric(seed_data, problem_type))
                time_values.append(seed_data.get('training_time_seconds', 0))
            
            if main_values:
                main_metrics.append(np.mean(main_values))
                secondary_metrics.append(np.mean(secondary_values))
                times.append(np.mean(time_values))
        
        if not main_metrics:
            print(f"    ⚠ Нет метрик для графика")
            continue
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Dataset: {dataset_name.upper()} - AutoKeras Results', fontsize=14, fontweight='bold')
        
        # График 1: Основная метрика
        ax1.plot(trials_list, main_metrics, 'b-', linewidth=2.5)
        ax1.set_xlabel('Number of Trials', fontsize=11)
        ax1.set_ylabel(main_metric_name, fontsize=11)
        ax1.set_title(f'{main_metric_name} vs Trials', fontsize=12)
        ax1.grid(True, alpha=0.2)
        
        # Добавляем значения на график
        for i, (trial, metric) in enumerate(zip(trials_list, main_metrics)):
            label = f'{metric:.3f}'
            ax1.annotate(label, xy=(trial, metric), xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom', fontsize=9)
        
        # График 2: Время обучения
        ax2.plot(trials_list, times, 'r-', linewidth=2.5)
        ax2.set_xlabel('Number of Trials', fontsize=11)
        ax2.set_ylabel('Training Time (seconds)', fontsize=11)
        ax2.set_title('Training Time vs Trials', fontsize=12)
        ax2.grid(True, alpha=0.2)
        
        # Добавляем значения времени
        for i, (trial, time_val) in enumerate(zip(trials_list, times)):
            minutes = time_val / 60
            label = f'{minutes:.1f} min'
            ax2.annotate(label, xy=(trial, time_val), xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Сохраняем график
        plot_filename = f"results/plots/{dataset_name}_autokeras.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ График сохранен: {plot_filename}")
        
        # Выводим таблицу результатов
        print(f"\n    📋 Результаты AutoKeras для {dataset_name}:")
        print(f"    {'Trials':<8} {main_metric_name:<15} {secondary_metric_name:<15} {'Time (min)':<12}")
        print(f"    {'-'*55}")
        for trial, main_metric, secondary_metric, time_val in zip(trials_list, main_metrics, secondary_metrics, times):
            print(f"    {trial:<8} {main_metric:<15.4f} {secondary_metric:<15.4f} {time_val/60:<12.1f}")

def plot_comparison_results(ak_results, neural_results):
    """Сравнивает AutoKeras с ручными архитектурами"""
    if not ak_results or not neural_results:
        print("\n❌ Нет данных для сравнения AutoKeras и ручных архитектур")
        return
    
    print("\n" + "="*80)
    print("СРАВНЕНИЕ AUTOKERAS И РУЧНЫХ АРХИТЕКТУР")
    print("="*80)
    
    os.makedirs("results/plots", exist_ok=True)
    
    for dataset_name in ak_results.keys():
        if dataset_name not in neural_results:
            continue
        
        print(f"\n  📊 Датасет: {dataset_name.upper()}")
        
        # Определяем тип задачи
        problem_type = get_problem_type(dataset_name, ak_results)
        main_metric_name = "R² Score" if problem_type == 'regression' else "Accuracy"
        
        # Данные для сравнения
        methods = []
        main_metrics = []
        times = []
        
        # AutoKeras (берем trials=60, seed=42)
        if 60 in ak_results[dataset_name] and '42' in ak_results[dataset_name][60]:
            ak_data = ak_results[dataset_name][60]['42']
            methods.append('AutoKeras (60)')
            main_metrics.append(get_main_metric(ak_data, problem_type))
            times.append(ak_data.get('training_time_seconds', 0))
        
        # Ручные архитектуры
        for arch_name in ['architecture1', 'architecture2']:
            if arch_name in neural_results[dataset_name]:
                arch_data = neural_results[dataset_name][arch_name]
                methods.append(f'Arch.{arch_name[-1]}')
                main_metrics.append(get_main_metric(arch_data, problem_type))
                times.append(arch_data.get('training_time_seconds', 0))
        
        if len(methods) < 2:
            print(f"    ⚠ Недостаточно данных для сравнения")
            continue
        
        # Создаем график сравнения
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Dataset: {dataset_name.upper()} - Comparison', fontsize=14, fontweight='bold')
        
        # График 1: Основная метрика
        colors = ['blue', 'orange', 'green'][:len(methods)]
        bars1 = ax1.bar(methods, main_metrics, color=colors, alpha=0.8)
        ax1.set_ylabel(main_metric_name, fontsize=11)
        ax1.set_title(f'{main_metric_name} Comparison', fontsize=12)
        ax1.grid(True, alpha=0.2, axis='y')
        
        # Добавляем значения на столбцы
        for bar, metric in zip(bars1, main_metrics):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{metric:.3f}', ha='center', va='bottom', fontsize=10)
        
        # График 2: Время обучения
        bars2 = ax2.bar(methods, times, color=colors, alpha=0.8)
        ax2.set_ylabel('Time (seconds)', fontsize=11)
        ax2.set_title('Training Time Comparison', fontsize=12)
        ax2.grid(True, alpha=0.2, axis='y')
        
        # Добавляем значения времени
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            minutes = time_val / 60
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{minutes:.1f} min', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Сохраняем график
        plot_filename = f"results/plots/{dataset_name}_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ График сравнения сохранен: {plot_filename}")
        
        # Выводим таблицу сравнения
        print(f"\n    📋 Сравнительная таблица для {dataset_name}:")
        print(f"    {'Method':<20} {main_metric_name:<15} {'Time (min)':<12}")
        print(f"    {'-'*50}")
        for method, metric, time_val in zip(methods, main_metrics, times):
            print(f"    {method:<20} {metric:<15.4f} {time_val/60:<12.1f}")

def print_summary_tables(ak_results):
    """Выводит сводные таблицы по результатам AutoKeras с учетом разных seeds"""
    if not ak_results:
        print("\n❌ Нет данных AutoKeras для сводных таблиц")
        return
    
    print("\n" + "="*80)
    print("СВОДНЫЕ ТАБЛИЦЫ (СРЕДНЕЕ ± СТАНДАРТНОЕ ОТКЛОНЕНИЕ)")
    print("="*80)
    
    for dataset_name, trials_data in ak_results.items():
        if not trials_data:
            continue
        
        print(f"\n📊 Датасет: {dataset_name.upper()}")
        
        # Определяем тип задачи
        problem_type = get_problem_type(dataset_name, ak_results)
        main_metric_name = "R² Score" if problem_type == 'regression' else "Accuracy"
        secondary_metric_name = "RMSE" if problem_type == 'regression' else "F1 Score"
        
        # Собираем все trials
        trials_list = sorted(trials_data.keys())
        
        # Краткая итоговая таблица
        print(f"\n    📋 ИТОГОВАЯ СВОДКА ДЛЯ {dataset_name.upper()}:")
        print(f"    {'Max Trials':<12} {main_metric_name:<15} {secondary_metric_name:<15} {'Время (мин)':<15}")
        print(f"    {'-'*60}")
        
        for trials in trials_list:
            seeds_data = trials_data[trials]
            
            if not seeds_data:
                continue
            
            # Вычисляем средние значения
            main_vals = [get_main_metric(d, problem_type) for d in seeds_data.values()]
            secondary_vals = [get_secondary_metric(d, problem_type) for d in seeds_data.values()]
            time_vals = [d.get('training_time_seconds', 0)/60 for d in seeds_data.values()]
            
            if main_vals:
                mean_main = np.mean(main_vals)
                mean_secondary = np.mean(secondary_vals)
                mean_time = np.mean(time_vals)
                
                # Форматируем вывод с отклонением
                if len(main_vals) > 1:
                    std_main = np.std(main_vals)
                    std_secondary = np.std(secondary_vals)
                    std_time = np.std(time_vals)
                    
                    print(f"    {trials:<12} {mean_main:.4f} ± {std_main:.4f}  {mean_secondary:.4f} ± {std_secondary:.4f}  {mean_time:.1f} ± {std_time:.1f}")
                else:
                    print(f"    {trials:<12} {mean_main:.4f}           {mean_secondary:.4f}           {mean_time:.1f}")

def print_neural_summary_tables(neural_results):
    """Выводит сводные таблицы для ручных архитектур"""
    if not neural_results:
        return
    
    print("\n" + "="*80)
    print("СВОДНЫЕ ТАБЛИЦЫ ДЛЯ РУЧНЫХ АРХИТЕКТУР")
    print("="*80)
    
    for dataset_name, architectures in neural_results.items():
        if not architectures:
            continue
        
        print(f"\n📊 Датасет: {dataset_name.upper()}")
        
        # Определяем тип задачи
        problem_type = get_problem_type(dataset_name, {})
        main_metric_name = "R² Score" if problem_type == 'regression' else "Accuracy"
        secondary_metric_name = "RMSE" if problem_type == 'regression' else "F1 Score"
        
        print(f"\n    Архитектура          {main_metric_name:<15} {secondary_metric_name:<15} {'Время (мин)':<15}")
        print(f"    {'-'*65}")
        
        for arch_name, arch_data in architectures.items():
            main_metric = get_main_metric(arch_data, problem_type)
            secondary_metric = get_secondary_metric(arch_data, problem_type)
            time_min = arch_data.get('training_time_seconds', 0) / 60
            
            print(f"    {arch_name:<20} {main_metric:<15.4f} {secondary_metric:<15.4f} {time_min:<15.1f}")

def main():
    """Главная функция визуализации"""
    print("="*80)
    print("📊 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print("="*80)
    print("Чтение данных напрямую из файловой структуры")
    print("="*80)
    
    # Создаем необходимые директории
    os.makedirs("results/plots", exist_ok=True)
    
    # Выводим архитектуры нейросетей
    print_architectures()
    
    # Загружаем результаты
    print("\n1. ЗАГРУЗКА ДАННЫХ:")
    print("-"*40)
    
    ak_results = load_autokeras_results()
    neural_results = load_neural_results()
    
    if not ak_results and not neural_results:
        print("\n❌ Нет данных для визуализации!")
        print("   Сначала запустите эксперименты:")
        print("   - Series A: python main.py (выберите 1)")
        print("   - Series B: python main.py (выберите 2)")
        return
    
    # Выводим сводные таблицы
    print("\n2. СВОДНЫЕ ТАБЛИЦЫ:")
    print("-"*40)
    
    if ak_results:
        print_summary_tables(ak_results)
    
    if neural_results:
        print_neural_summary_tables(neural_results)
    
    # Строим графики
    print("\n3. СОЗДАНИЕ ГРАФИКОВ:")
    print("-"*40)
    
    if ak_results:
        plot_autokeras_results(ak_results)
    
    if ak_results and neural_results:
        plot_comparison_results(ak_results, neural_results)
    
    print("\n" + "="*80)
    print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("📁 Все графики сохранены в папке: results/plots/")
    print("="*80)

if __name__ == "__main__":
    main()