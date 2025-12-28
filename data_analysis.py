import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Загрузка и анализ датасетов
def load_and_analyze_datasets():
    datasets = {
        'bank': 'data/raw/Bank Marketing.csv',
        'housing': 'data/raw/housing.csv',
        'churn': 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    }
    loaded_data = {}
    for name, path in datasets.items():
        print(f"\nДатасет: {name}")
        df = pd.read_csv(path)
        loaded_data[name] = df
        
        print(f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
        
        print(f"\nОписательная статистика (числовые признаки):")
        print(df.describe())
        
        print(f"\nОписательная статистика (категориальные признаки):")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            # Используем встроенную функцию describe() для категориальных данных
            cat_stats = df[categorical_cols].describe()
            print(cat_stats)
            # Дополнительная статистика для категориальных признаков
            print(f"\nДополнительная статистика категориальных данных:")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
                mode_freq = (df[col] == mode_value).sum()
                mode_percent = (mode_freq / len(df)) * 100
                print(f"{col}:")
                print(f"Уникальных значений: {unique_count}")
                print(f"Самое частое: '{mode_value}' ({mode_freq} раз, {mode_percent:.1f}%)")
                if unique_count <= 10:  # Показываем все значения, если их мало
                    print(f"    • Все значения: {', '.join([str(v) for v in df[col].unique()[:10]])}")
                elif unique_count > 10:
                    print(f"    • Топ-5 значений: {df[col].value_counts().head().to_dict()}")
        else:
            print("Категориальные признаки отсутствуют")
        
        print(f"\nТипы данных:")
        print(df.dtypes.value_counts())
        
        print(f"\nПропущенные значения:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print(missing)
        else:
            print("Пропущенных значений нет")
    
    return loaded_data

# Визуализация исходных данных
def visualize_raw_data(datasets):
    for name, df in datasets.items():
        print(f"\nВизуализация {name}")

        # Создаем графики
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Dataset: {name.upper()} - Raw Data Analysis', fontsize=16, fontweight='bold')
        
        # Распределение числовых признаков (первые 3)
        ax = axes[0, 0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Берем первые 3 числовых признака
            for i, col in enumerate(numeric_cols[:3]):
                ax.hist(df[col].dropna(), bins=30, alpha=0.6, label=col[:15])
            ax.set_title('Распределение числовых признаков', fontsize=12)
            ax.set_xlabel('Значение', fontsize=10)
            ax.set_ylabel('Частота', fontsize=10)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Нет числовых признаков', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Матрица корреляций
        ax = axes[0, 1]
        corr_matrix = df[numeric_cols].corr()
        
        # Ограничиваем количество признаков для читаемости
        if len(corr_matrix) > 10:
            corr_matrix = corr_matrix.iloc[:10, :10]
        
        # Создаем тепловую карту
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Корреляционная матрица', fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Категориальные признаки
        ax = axes[1, 1]
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            # Берем первый категориальный признак
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(8)
            colors = plt.cm.Set3(np.arange(len(value_counts)))
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, color=colors)
            ax.set_title(f'Категориальный признак: {col}', fontsize=12)
            ax.set_ylabel('Количество', fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            
            # Добавляем значения над столбцами
            for bar, count in zip(bars, value_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{count}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Нет категориальных признаков', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/data analysis/{name}_raw_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранен: results/data analysis/{name}_raw_analysis.png")
        
        # Выводим ключевую статистику
        print(f"\nРаспределение переменных {name}:")
        
        # Статистика для числовых признаков
        if numeric_cols:
            print(f"Числовые признаки ({len(numeric_cols)}):")
            for col in numeric_cols[:5]:  # Только первые 5 для краткости
                stats = df[col].describe()
                print(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

# Визуализация целевых переменных
def visualize_target_variables(datasets):
    loaded_data = {}
    # Названия целевых переменных для каждого датасета
    target_columns = {
        'bank': 'Subscription',
        'housing': 'median_house_value',
        'churn': 'Churn'
    }
    print('ak;fk;alkfd =\n', datasets)
    for name, df in datasets.items():
        loaded_data[name] = {
            'data': df,
            'target': target_columns.get(name)
        }
        print(f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")

    for name, dataset_info in loaded_data.items():
        df = dataset_info['data']
        target_col = dataset_info['target']
            
        print(f"\nВизуализация целевой переменной для {name}")
        
        # Проверяем наличие целевой переменной
        if target_col not in df.columns:
            print(f"Целевая переменная '{target_col}' не найдена в датасете {name}")
            continue
        
        # Создаем график
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Dataset: {name.upper()} - Target Variable: {target_col}', 
                    fontsize=14, fontweight='bold')
        
        # Определяем тип переменной
        if target_col in ['median_house_value']:
            # Числовая переменная
            ax1 = axes[0]
            ax1.hist(df[target_col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Распределение {target_col}', fontsize=12)
            ax1.set_xlabel('Значение', fontsize=10)
            ax1.set_ylabel('Частота', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Боксплот
            ax2 = axes[1]
            ax2.boxplot(df[target_col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen'))
            ax2.set_title(f'Боксплот {target_col}', fontsize=12)
            ax2.set_ylabel('Значение', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Статистика
            stats = df[target_col].describe()
            print(f"Статистика {target_col}:")
            print(f"  Количество: {stats['count']:.0f}")
            print(f"  Среднее: {stats['mean']:.2f}")
            print(f"  Стандартное отклонение: {stats['std']:.2f}")
            print(f"  Минимум: {stats['min']:.2f}")
            print(f"  25%: {stats['25%']:.2f}")
            print(f"  Медиана: {stats['50%']:.2f}")
            print(f"  75%: {stats['75%']:.2f}")
            print(f"  Максимум: {stats['max']:.2f}")
            
        else:
            # Категориальная переменная
            ax1 = axes[0]
            value_counts = df[target_col].value_counts()
            colors = plt.cm.Set3(np.arange(len(value_counts)))
            bars = ax1.bar(range(len(value_counts)), value_counts.values, color=colors)
            ax1.set_title(f'Распределение {target_col}', fontsize=12)
            ax1.set_xlabel('Категория', fontsize=10)
            ax1.set_ylabel('Количество', fontsize=10)
            ax1.set_xticks(range(len(value_counts)))
            ax1.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Добавляем значения над столбцами
            for bar, count in zip(bars, value_counts.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{count}', ha='center', va='bottom', fontsize=9)
            
            # Круговая диаграмма
            ax2 = axes[1]
            wedges, texts, autotexts = ax2.pie(value_counts.values, 
                                               labels=value_counts.index,
                                               autopct='%1.1f%%',
                                               startangle=90,
                                               colors=colors)
            ax2.set_title(f'Доли категорий {target_col}', fontsize=12)
            
            # Статистика
            print(f"Статистика {target_col}:")
            print(f"  Уникальных значений: {len(value_counts)}")
            for idx, (value, count) in enumerate(value_counts.items()):
                percentage = (count / len(df)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
        
        plt.tight_layout()
        plt.savefig(f'results/data analysis/{name}_target_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"График сохранен: results/data analysis/{name}_target_analysis.png")

def main():
    
    # Создаем папку для результатов
    os.makedirs("results", exist_ok=True)
    
    # 1. Загрузка и анализ
    datasets = load_and_analyze_datasets()
    visualize_target_variables(datasets)

    # 2. Визуализация исходных данных
    visualize_raw_data(datasets)
    
    print("\nРезультаты:")
    print("1. Графики анализа: results/*_analysis.png")
    print("2. Матрицы корреляций: results/*_correlation_matrix.png")
    print("3. Обработанные данные: results/*_processed.csv")
    print("4. Отчет: results/summary_report.txt")

if __name__ == "__main__":
    main()