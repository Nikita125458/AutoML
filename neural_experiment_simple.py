import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import os
import json
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import random
import warnings

# ========== ОТКЛЮЧЕНИЕ WARNING'ОВ ==========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Отключаем warnings Python
warnings.filterwarnings('ignore')

# Отключаем TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Отключаем deprecated warnings для совместимости
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# ===========================================
# Эксперимент с двумя архитектурами
class SimpleNeuralExperiment:
    def __init__(self, X_train, y_train, X_test, y_test, problem_type, dataset_name):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.problem_type = problem_type
        self.dataset_name = dataset_name
        self.input_dim = X_train.shape[1]
    
    # Архитектура 1    
    def create_architecture1(self, input_shape):
        print("Создание архитектуры 1")
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        model.add(keras.layers.Dense(256, activation='relu', kernel_initializer='random_uniform'))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        
        # Выходной слой
        if self.problem_type == 'classification':
            num_classes = len(np.unique(self.y_train))
            if num_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(keras.layers.Dense(num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            model.add(keras.layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        # оптимизатор
        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.995)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        print(f"  Всего параметров: {model.count_params():,}")
        
        return model
    
    # Архитектура 2
    def create_architecture2(self, input_shape):
        print("Создание архитектуры 2")
        
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        layers_config = [(64, 'relu', 0.2), (32, 'tanh', 0.3), (128, 'relu', 0.1), (64, 'sigmoid', 0.4), (256, 'relu', 0.2), (128, 'tanh', 0.3), (64, 'relu', 0.1),(32, 'sigmoid', 0.4),]
        for i, (units, activation, dropout) in enumerate(layers_config):
            model.add(keras.layers.Dense(units, activation=activation))
            model.add(keras.layers.Dropout(dropout))
        
        # Выходной слой
        if self.problem_type == 'classification':
            num_classes = len(np.unique(self.y_train))
            if num_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(keras.layers.Dense(num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            model.add(keras.layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        # оптимизатор
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"  Всего параметров: {model.count_params():,}")
        
        return model
    
    # обучение 
    def train_neural_model(self, model, time_limit_seconds=3600):
        print("Обучение модели")
        start_time = time.time()
        
        # Callback для отслеживания времени
        class TimeLimitCallback(keras.callbacks.Callback):
            def __init__(self, time_limit, start_time):
                self.time_limit = time_limit
                self.start_time = start_time
                
            def on_epoch_end():
                current_time = time.time()
                elapsed = current_time - self.start_time
                if elapsed > self.time_limit:
                    self.model.stop_training = True
                    print(f"\nДостигнут лимит времени: {elapsed:.1f} секунд")
        
        time_callback = TimeLimitCallback(time_limit_seconds, start_time)
        
        # Использование критерия ранней остановки при выходе метрики на плато
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001, verbose=1)
        
        # Обучение
        history = model.fit(self.X_train, self.y_train, validation_split=0.2, epochs=200, batch_size=16, callbacks=[early_stop, time_callback], verbose=1)
        training_time = time.time() - start_time
        
        # Оценка на тесте
        if self.problem_type == 'classification':
            y_pred = model.predict(self.X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(self.y_test, y_pred_classes)
            f1 = f1_score(self.y_test, y_pred_classes, average='weighted')
            
            test_metrics = {'accuracy': float(accuracy),'f1_score': float(f1)}
        else:
            y_pred = model.predict(self.X_test, verbose=0)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            test_metrics = {'mse': float(mse), 'rmse': float(rmse), 'r2_score': float(r2)}
        
        return model, test_metrics, training_time, history
    
    # запуск эксперимента для одной архитектуры
    def run_architecture_experiment(self, architecture_name, time_limit_seconds=3600):
        print(f"Лимит времени: {time_limit_seconds//60} минут")
        
        input_shape = (self.input_dim,)
        
        # создание модели
        if architecture_name == 'architecture1':
            model = self.create_architecture1(input_shape)
        else:
            model = self.create_architecture2(input_shape)
        
        # обучение
        trained_model, test_metrics, training_time, history = self.train_neural_model(model, architecture_name, time_limit_seconds)
        
        # сохранение модели и результатов
        models_dir = f"models/neural/{self.dataset_name}"
        results_dir = f"results/neural/{self.dataset_name}"
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # сохранение модели
        model_filename = f"{models_dir}/{architecture_name}.keras"
        trained_model.save(model_filename)
        
        # анализ истории обучения
        if 'val_loss' in history.history and len(history.history['val_loss']) > 0:
            final_val_loss = history.history['val_loss'][-1]
            best_val_loss = min(history.history['val_loss'])
            overfitting_degree = (history.history['loss'][-1] / final_val_loss) if final_val_loss > 0 else 1.0
        else:
            final_val_loss = best_val_loss = overfitting_degree = 0
        
        # подготовка результатов
        results = {
            'dataset': self.dataset_name,
            'problem_type': self.problem_type,
            'architecture': architecture_name,
            'time_limit_seconds': time_limit_seconds,
            'training_time_seconds': float(training_time),
            'test_metrics': test_metrics,
            'param_count': int(trained_model.count_params()),
            'model_path': model_filename,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_history': {
                'loss': [float(x) for x in history.history.get('loss', [])],
                'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                'accuracy': [float(x) for x in history.history.get('accuracy', [])] if 'accuracy' in history.history else [],
                'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])] if 'val_accuracy' in history.history else [],
                'final_val_loss': float(final_val_loss),
                'best_val_loss': float(best_val_loss),
                'overfitting_degree': float(overfitting_degree)
            }
        }
        
        # сохранение результатов
        results_filename = f"{results_dir}/{architecture_name}_results.json"
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        # вывод краткой сводки
        print(f"\nРезультаты {architecture_name}:")
        print(f"  Время обучения: {training_time:.2f} сек ({training_time/60:.1f} мин)")
        
        if self.problem_type == 'classification':
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  F1: {test_metrics['f1_score']:.4f}")
        else:
            print(f"  R²: {test_metrics['r2_score']:.4f}")
            print(f"  RMSE: {test_metrics['rmse']:.4f}")
        
        print(f"  Параметров модели: {trained_model.count_params():,}")
        print(f"  Эпох обучено: {len(history.history.get('loss', []))}")
        
        if 'val_loss' in history.history:
            print(f"  Лучшая val_loss: {best_val_loss:.4f}")
            if overfitting_degree > 1.5:
                print(f"  Возможное переобучение: loss/val_loss = {overfitting_degree:.2f}")
        
        print(f"  Результаты сохранены в: {results_filename}")
        
        return results, trained_model
    
    # запуск эксперимента для обеих архитектур
    def run_both_architectures(self):
        print(f"Серия B: Сравнение AutoKeras с двумя ручными архитектурами")
        print(f"Датасет: {self.dataset_name}")
        print(f"Тип задачи: {self.problem_type}")
        
        all_results = {}
        
        # Архитектура 1 (1 час)
        print(f"\nАрхитектура 1")
        results1, model1 = self.run_architecture_experiment('architecture1', 3600)
        all_results['architecture1'] = results1
        
        # Архитектура 2 (1 час)
        print(f"\nАрхитектура 2")
        results2, model2 = self.run_architecture_experiment('architecture2', 3600)
        all_results['architecture2'] = results2
        
        print(f"\nОбе ручные архитектуры обучены")
        
        return all_results, [model1, model2]