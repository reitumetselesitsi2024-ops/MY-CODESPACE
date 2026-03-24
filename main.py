from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import json
import os
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
import random
import hashlib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten, Conv1D, MaxPooling1D, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# sklearn models (all work in Replit)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ============= CONFIGURATION =============
SCRAPE_INTERVAL_MINUTES = 15
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 30
MAX_CONSECUTIVE_FAILURES = 5
DUPLICATE_TIME_THRESHOLD_HOURS = 2
JSON_FILENAME = "results.json"
# ==========================================

# Color mapping for numbers 1-48
COLOR_MAP = {
    'red': [1, 9, 17, 25, 33, 41],
    'green': [2, 10, 18, 26, 34, 42],
    'blue': [3, 11, 19, 27, 35, 43],
    'pink': [4, 12, 20, 28, 36, 44],
    'brown': [5, 13, 21, 29, 37, 45],
    'yellow': [6, 14, 22, 30, 38, 46],
    'orange': [7, 15, 23, 31, 39, 47],
    'black': [8, 16, 24, 32, 40, 48]
}

NUMBER_TO_COLOR = {}
for color, numbers in COLOR_MAP.items():
    for num in numbers:
        NUMBER_TO_COLOR[num] = color

class UltimatePredictor:
    """Maximum Power Lottery Predictor for Replit"""
    
    def __init__(self):
        # Deep Learning Models
        self.lstm_model = None
        self.bilstm_model = None
        self.gru_model = None
        self.cnn_lstm_model = None
        
        # Traditional ML Models
        self.rf_model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
        self.gb_model = GradientBoostingClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, random_state=42)
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), max_iter=500, random_state=42, early_stopping=True)
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Scaling
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Clustering
        self.kmeans = KMeans(n_clusters=8, random_state=42)
        
        self.is_trained = False
        self.sequence_length = 20
        
    def extract_features(self, numbers):
        """Extract maximum features from a draw"""
        if not numbers:
            return []
        
        numbers = sorted([int(n) for n in numbers])
        features = []
        
        # Statistical features
        features.append(float(np.mean(numbers)))
        features.append(float(np.std(numbers)))
        features.append(float(np.min(numbers)))
        features.append(float(np.max(numbers)))
        features.append(float(np.median(numbers)))
        features.append(float(np.percentile(numbers, 25)))
        features.append(float(np.percentile(numbers, 75)))
        features.append(float(np.percentile(numbers, 90)))
        features.append(float(np.percentile(numbers, 10)))
        
        # Gap analysis
        gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        if gaps:
            features.append(float(np.mean(gaps)))
            features.append(float(np.std(gaps)))
            features.append(float(np.max(gaps)))
            features.append(float(np.min(gaps)))
            features.append(float(np.median(gaps)))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Sum statistics
        features.append(float(sum(numbers)))
        features.append(float(sum(n**2 for n in numbers)))
        
        # Color distribution
        colors = [NUMBER_TO_COLOR.get(n, 'unknown') for n in numbers]
        for color in COLOR_MAP.keys():
            features.append(float(colors.count(color)))
        
        # Odd/Even analysis
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        features.append(float(odd_count))
        features.append(float(6 - odd_count))
        features.append(float(odd_count / 6))
        
        # Prime numbers
        primes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}
        prime_count = sum(1 for n in numbers if n in primes)
        features.append(float(prime_count))
        
        # Fibonacci numbers
        fib = {1,2,3,5,8,13,21,34,55}
        fib_count = sum(1 for n in numbers if n in fib)
        features.append(float(fib_count))
        
        # Range
        features.append(float(max(numbers) - min(numbers)))
        
        # Pattern detection
        arith_prog = 0
        for i in range(len(numbers)-2):
            if numbers[i+1] - numbers[i] == numbers[i+2] - numbers[i+1]:
                arith_prog += 1
        features.append(float(arith_prog))
        
        # Consecutive numbers
        consec = sum(1 for i in range(len(numbers)-1) if numbers[i+1] - numbers[i] == 1)
        features.append(float(consec))
        
        # Clustering feature
        mean_pos = float(np.mean(numbers))
        cluster_dist = float(sum(abs(n - mean_pos) for n in numbers) / 6)
        features.append(cluster_dist)
        
        # Entropy
        unique, counts = np.unique(numbers, return_counts=True)
        probs = counts / len(numbers)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features.append(float(entropy))
        
        return features
    
    def build_lstm_model(self):
        """Build advanced LSTM"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 48)),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(48, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def build_bilstm_model(self):
        """Build Bidirectional LSTM"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.sequence_length, 48)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(48, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def build_cnn_lstm_model(self):
        """Build CNN-LSTM hybrid"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(self.sequence_length, 48)),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(48, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def prepare_sequence_data(self, results):
        """Prepare data for deep learning"""
        draw_sequences = []
        for result in results:
            numbers = sorted([int(n) for n in result.get('first_draw_numbers', [])])
            one_hot = np.zeros(48)
            for num in numbers:
                one_hot[num-1] = 1
            draw_sequences.append(one_hot)
        
        X, y = [], []
        for i in range(len(draw_sequences) - self.sequence_length):
            X.append(draw_sequences[i:i + self.sequence_length])
            y.append(draw_sequences[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train_deep_learning(self, results):
        """Train deep learning models"""
        if len(results) < self.sequence_length + 10:
            return False
        
        print("   Training Deep Learning Models...")
        X, y = self.prepare_sequence_data(results)
        
        if len(X) < 30:
            return False
        
        X_scaled = self.minmax_scaler.fit_transform(X.reshape(-1, 48)).reshape(-1, self.sequence_length, 48)
        
        split = int(len(X) * 0.8)
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        
        try:
            self.lstm_model = self.build_lstm_model()
            self.lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)
            print("      ✓ LSTM trained")
        except:
            print("      ✗ LSTM training failed")
        
        try:
            self.bilstm_model = self.build_bilstm_model()
            self.bilstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)
            print("      ✓ BiLSTM trained")
        except:
            print("      ✗ BiLSTM training failed")
        
        try:
            self.cnn_lstm_model = self.build_cnn_lstm_model()
            self.cnn_lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=0)
            print("      ✓ CNN-LSTM trained")
        except:
            print("      ✗ CNN-LSTM training failed")
        
        return True
    
    def train_traditional_ml(self, results):
        """Train traditional ML models"""
        print("   Training Traditional ML Models...")
        
        X, y = [], []
        for i in range(len(results) - 1):
            current = results[i].get('first_draw_numbers', [])
            next_draw = results[i+1].get('first_draw_numbers', [])
            
            if len(current) == 6 and len(next_draw) == 6:
                features = self.extract_features(current)
                if len(features) > 0:
                    X.append(features)
                    target = np.zeros(48)
                    for num in [int(n) for n in next_draw]:
                        target[num-1] = 1
                    y.append(target)
        
        if len(X) < 30:
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        
        try:
            self.rf_model.fit(X_scaled, y)
            print("      ✓ Random Forest trained")
        except:
            print("      ✗ Random Forest failed")
        
        try:
            self.gb_model.fit(X_scaled, y)
            print("      ✓ Gradient Boosting trained")
        except:
            print("      ✗ Gradient Boosting failed")
        
        try:
            self.mlp_model.fit(X_scaled, y)
            print("      ✓ Neural Network trained")
        except:
            print("      ✗ Neural Network failed")
        
        try:
            self.svm_model.fit(X_scaled, y)
            print("      ✓ SVM trained")
        except:
            print("      ✗ SVM failed")
        
        # KMeans clustering
        try:
            self.kmeans.fit(X_scaled)
            print("      ✓ KMeans clustering trained")
        except:
            pass
        
        return True
    
    def train(self, results):
        """Train ALL models"""
        print("\n" + "=" * 80)
        print("🧠 TRAINING POWERFUL AI MODELS...")
        print("=" * 80)
        
        if len(results) < 30:
            print(f"⚠️ Need 30+ rounds for training. Currently: {len(results)}")
            return False
        
        self.train_deep_learning(results)
        self.train_traditional_ml(results)
        
        self.is_trained = True
        print("\n✅ MODELS TRAINED SUCCESSFULLY!")
        print("   • Deep Learning: LSTM, BiLSTM, CNN-LSTM")
        print("   • Traditional ML: Random Forest, Gradient Boosting, Neural Network, SVM")
        print("   • Clustering: KMeans")
        print("=" * 80)
        return True
    
    def predict_with_deep_learning(self, recent_results):
        """Get predictions from deep learning models"""
        if len(recent_results) < self.sequence_length:
            return Counter()
        
        predictions = Counter()
        
        # Prepare sequence
        draw_sequence = []
        for result in recent_results[-self.sequence_length:]:
            numbers = sorted([int(n) for n in result.get('first_draw_numbers', [])])
            one_hot = np.zeros(48)
            for num in numbers:
                one_hot[num-1] = 1
            draw_sequence.append(one_hot)
        
        X = np.array([draw_sequence])
        X_scaled = self.minmax_scaler.transform(X.reshape(-1, 48)).reshape(-1, self.sequence_length, 48)
        
        if self.lstm_model:
            try:
                lstm_pred = self.lstm_model.predict(X, verbose=0)[0]
                for i, prob in enumerate(lstm_pred):
                    predictions[i+1] += float(prob) * 1.3
            except:
                pass
        
        if self.bilstm_model:
            try:
                bilstm_pred = self.bilstm_model.predict(X, verbose=0)[0]
                for i, prob in enumerate(bilstm_pred):
                    predictions[i+1] += float(prob) * 1.4
            except:
                pass
        
        if self.cnn_lstm_model:
            try:
                cnn_pred = self.cnn_lstm_model.predict(X, verbose=0)[0]
                for i, prob in enumerate(cnn_pred):
                    predictions[i+1] += float(prob) * 1.3
            except:
                pass
        
        return predictions
    
    def predict_with_traditional_ml(self, last_draw):
        """Get predictions from traditional ML models"""
        features = self.extract_features(last_draw)
        if len(features) == 0:
            return Counter()
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        predictions = Counter()
        
        try:
            rf_pred = self.rf_model.predict_proba(X_scaled)[0]
            for i, prob in enumerate(rf_pred):
                predictions[i+1] += float(prob) * 1.3
        except:
            pass
        
        try:
            gb_pred = self.gb_model.predict_proba(X_scaled)[0]
            for i, prob in enumerate(gb_pred):
                predictions[i+1] += float(prob) * 1.4
        except:
            pass
        
        try:
            mlp_pred = self.mlp_model.predict_proba(X_scaled)[0]
            for i, prob in enumerate(mlp_pred):
                predictions[i+1] += float(prob) * 1.2
        except:
            pass
        
        try:
            svm_pred = self.svm_model.predict_proba(X_scaled)[0]
            for i, prob in enumerate(svm_pred):
                predictions[i+1] += float(prob) * 1.1
        except:
            pass
        
        return predictions
    
    def ensemble_predict(self, results, last_draw):
        """Combine ALL prediction methods"""
        print("\n🎯 GENERATING ENSEMBLE PREDICTION...")
        
        all_predictions = Counter()
        
        # Deep Learning predictions
        dl_pred = self.predict_with_deep_learning(results)
        for num, score in dl_pred.items():
            all_predictions[num] += score
        
        # Traditional ML predictions
        ml_pred = self.predict_with_traditional_ml(last_draw)
        for num, score in ml_pred.items():
            all_predictions[num] += score
        
        # Get top 12 numbers
        sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        top_12 = [int(num) for num, _ in sorted_preds[:12]]
        
        # Ensure we have 12 numbers
        if len(top_12) < 12:
            freq = Counter()
            for result in results[-100:]:
                numbers = [int(n) for n in result.get('first_draw_numbers', [])]
                freq.update(numbers)
            hot = [int(num) for num, _ in freq.most_common(30)]
            for num in hot:
                if num not in top_12:
                    top_12.append(num)
                if len(top_12) >= 12:
                    break
        
        print(f"   Ensemble complete! Top 12 numbers: {sorted(top_12[:12])}")
        
        return top_12[:6], top_12[6:12]

def extract_numbers_from_balls(balls_div):
    """Extract numbers from balls container"""
    numbers = []
    buttons = balls_div.find_elements(By.TAG_NAME, "button")
    for button in buttons:
        text = button.text.strip()
        if text and text.isdigit():
            numbers.append(text)
    return numbers

def load_existing_data():
    """Load existing data from JSON file"""
    if os.path.exists(JSON_FILENAME):
        try:
            with open(JSON_FILENAME, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('results', [])
        except Exception as e:
            print(f"⚠️ Error loading file: {e}")
    return []

def save_results(results):
    """Save results to JSON file"""
    results.sort(key=lambda x: x.get('round_number', 0), reverse=True)
    json_data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": len(results),
        "results": results
    }
    with open(JSON_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    return len(results)

def calculate_confidence(results, prediction):
    """Calculate confidence based on historical accuracy"""
    if len(results) < 20:
        return 50.0
    
    pred_set = set([int(n) for n in prediction])
    hits = 0
    total = 0
    
    for i in range(len(results) - 1):
        actual = set([int(n) for n in results[i+1].get('first_draw_numbers', [])])
        overlap = len(pred_set & actual)
        hits += overlap
        total += 6
    
    return float((hits / total) * 100)

def get_next_round_number(results):
    """Get next round number"""
    if not results:
        return 1
    latest = max(int(r.get('round_number', 0)) for r in results)
    return int(latest + 1)

def predict_next_5_rounds(results, predictor, last_main, last_first):
    """Predict next 5 rounds"""
    next_round_num = get_next_round_number(results)
    
    predictions = []
    for i in range(5):
        round_num = next_round_num + i
        
        var_main = list(last_main)
        var_first = list(last_first)
        
        if i > 0:
            shift = i * 2
            var_main = [(n + shift) % 48 + 1 if n + shift > 48 else n + shift for n in var_main]
            var_first = [(n + shift) % 48 + 1 if n + shift > 48 else n + shift for n in var_first]
        
        var_main = list(set(var_main))
        var_first = list(set(var_first))
        
        freq = Counter()
        for result in results[-100:]:
            numbers = [int(n) for n in result.get('first_draw_numbers', [])]
            freq.update(numbers)
        hot = [int(num) for num, _ in freq.most_common(30)]
        
        while len(var_main) < 6:
            for num in hot:
                if num not in var_main:
                    var_main.append(num)
                    break
        
        while len(var_first) < 6:
            for num in hot:
                if num not in var_first:
                    var_first.append(num)
                    break
        
        var_main = [int(n) for n in var_main]
        var_first = [int(n) for n in var_first]
        
        predictions.append({
            'round_number': int(round_num),
            'main_numbers': sorted(var_main[:6]),
            'first_draw_numbers': sorted(var_first[:6]),
            'confidence': float(round(calculate_confidence(results, var_main[:6]), 1))
        })
    
    return predictions

def scrape_data(driver):
    """Scrape all lottery data from website"""
    print("\n📡 SCRAPING DATA...")
    
    driver.get('https://www.simacombet.com/luckysix')
    time.sleep(3)
    
    iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "PluginLuckySix"))
    )
    driver.switch_to.frame(iframe)
    
    button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Results')]"))
    )
    button.click()
    print("✅ Results button clicked")
    time.sleep(3)
    
    round_rows = driver.find_elements(By.CSS_SELECTOR, "div.round-row")
    print(f"✅ Found {len(round_rows)} rounds")
    
    existing_results = load_existing_data()
    existing_round_nums = {r.get('round_number'): r for r in existing_results}
    new_results = []
    
    for i in range(len(round_rows) - 1, -1, -1):
        rows = driver.find_elements(By.CSS_SELECTOR, "div.round-row")
        current_row = rows[i]
        
        title_element = current_row.find_element(By.CSS_SELECTOR, "div.accordion-title")
        title_text = title_element.text.strip()
        round_num = re.search(r'Round\s*(\d+)', title_text)
        round_num = int(round_num.group(1)) if round_num else None
        
        if round_num in existing_round_nums:
            continue
        
        driver.execute_script("arguments[0].scrollIntoView();", current_row)
        time.sleep(0.5)
        current_row.click()
        time.sleep(2)
        
        try:
            draw_sequences = driver.find_elements(By.CSS_SELECTOR, "div.draw-sequence")
            first_draw_numbers = []
            second_draw_numbers = []
            
            for seq in draw_sequences:
                seq_title = seq.find_element(By.CSS_SELECTOR, "div.title").text.lower()
                balls_containers = seq.find_elements(By.CSS_SELECTOR, "div.balls")
                
                if "drawn" in seq_title:
                    for container in balls_containers:
                        first_draw_numbers.extend(extract_numbers_from_balls(container))
                elif "bonus" in seq_title:
                    if balls_containers:
                        second_draw_numbers = extract_numbers_from_balls(balls_containers[0])
            
            result = {
                'round_number': int(round_num),
                'round_title': title_text,
                'first_draw_numbers': [int(n) for n in first_draw_numbers],
                'second_draw_numbers': [int(n) for n in second_draw_numbers],
                'timestamp': datetime.now().isoformat()
            }
            new_results.append(result)
            print(f"✅ Round {round_num} saved")
            
        except Exception as e:
            print(f"❌ Error on Round {round_num}: {e}")
        
        current_row.click()
        time.sleep(1)
    
    all_results = new_results + existing_results
    total = save_results(all_results)
    
    print(f"\n💾 Total rounds: {total}")
    if new_results:
        print(f"   New rounds added: {len(new_results)}")
    
    return all_results

def run_prediction(results):
    """Run complete prediction analysis"""
    if len(results) < 10:
        print(f"⚠️ Only {len(results)} rounds. Need at least 10 for predictions.")
        return
    
    print("\n" + "=" * 80)
    print("🤖 POWERFUL AI LOTTERY PREDICTION SYSTEM")
    print("=" * 80)
    print(f"📊 Analyzing {len(results)} rounds of historical data...")
    print("=" * 80)
    
    predictor = UltimatePredictor()
    
    if len(results) >= 30:
        predictor.train(results)
    else:
        print(f"⚠️ Need 30+ rounds for full training. Currently: {len(results)}")
    
    # Basic statistics
    freq = Counter()
    color_freq = Counter()
    for result in results:
        numbers = [int(n) for n in result.get('first_draw_numbers', [])]
        freq.update(numbers)
        for num in numbers:
            color_freq[NUMBER_TO_COLOR.get(num, 'unknown')] += 1
    
    hot = [int(num) for num, _ in freq.most_common(10)]
    cold = [int(num) for num, _ in sorted(freq.items(), key=lambda x: x[1])[:10]]
    
    print(f"\n🔥 HOT NUMBERS: {hot}")
    print(f"❄️ COLD NUMBERS: {cold}")
    
    print(f"\n🎨 COLOR DISTRIBUTION:")
    for color, count in sorted(color_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"   {color}: {count}")
    
    # Get predictions
    last_draw = [int(n) for n in results[-1].get('first_draw_numbers', [])]
    
    if predictor.is_trained:
        main, first = predictor.ensemble_predict(results, last_draw)
    else:
        main = hot[:6]
        first = hot[6:12] if len(hot) >= 12 else hot[:6]
    
    main = [int(n) for n in main]
    first = [int(n) for n in first]
    confidence = calculate_confidence(results, main)
    next_round_num = get_next_round_number(results)
    
    print(f"\n" + "=" * 80)
    print(f"🎯 AI PREDICTION FOR ROUND {next_round_num}")
    print("=" * 80)
    print(f"   6 numbers that will appear:     {sorted(main)}")
    print(f"   6 numbers that will appear first: {sorted(first)}")
    print(f"   Confidence: {confidence:.1f}%")
    
    print(f"\n" + "=" * 80)
    print(f"🔮 PREDICTIONS FOR NEXT 5 ROUNDS")
    print("=" * 80)
    
    next_5 = predict_next_5_rounds(results, predictor, main, first)
    
    for pred in next_5:
        print(f"\n   ROUND {pred['round_number']}:")
        print(f"      6 numbers that will appear:     {pred['main_numbers']}")
        print(f"      6 numbers that will appear first: {pred['first_draw_numbers']}")
        print(f"      Confidence: {pred['confidence']}%")
    
    # Save predictions
    save_data = {
        'generated': datetime.now().isoformat(),
        'total_rounds_analyzed': int(len(results)),
        'hot_numbers': [int(n) for n in hot],
        'cold_numbers': [int(n) for n in cold],
        'color_distribution': {str(k): int(v) for k, v in dict(color_freq).items()},
        'models_used': {
            'deep_learning': ['LSTM', 'BiLSTM', 'CNN-LSTM'] if predictor.is_trained else [],
            'traditional_ml': ['RandomForest', 'GradientBoosting', 'NeuralNetwork', 'SVM'] if predictor.is_trained else [],
            'clustering': ['KMeans'] if predictor.is_trained else []
        },
        'next_round': {
            'round_number': int(next_round_num),
            'main_numbers': [int(n) for n in sorted(main)],
            'first_draw_numbers': [int(n) for n in sorted(first)],
            'confidence': float(confidence)
        },
        'next_5_rounds': next_5
    }
    
    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Predictions saved to predictions.json")

def perform_scrape_and_predict():
    """Perform scrape and immediately run predictions"""
    driver = None
    
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        
        driver = webdriver.Chrome(options=options)
        print("✅ Chrome ready")
        
        results = scrape_data(driver)
        run_prediction(results)
        
        return True, len(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0
        
    finally:
        if driver:
            driver.quit()

def run_scraping_loop():
    """Main scraping loop with retry logic"""
    consecutive_failures = 0
    iteration = 0
    
    print("=" * 80)
    print("🤖 POWERFUL AI LOTTERY PREDICTION SYSTEM")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Scrape interval: {SCRAPE_INTERVAL_MINUTES} minutes")
    print("")
    print("🧠 DEEP LEARNING MODELS:")
    print("   • LSTM - Long Short-Term Memory")
    print("   • BiLSTM - Bidirectional LSTM")
    print("   • CNN-LSTM - Convolutional + LSTM")
    print("")
    print("📊 TRADITIONAL ML MODELS:")
    print("   • Random Forest (500 trees)")
    print("   • Gradient Boosting (300 estimators)")
    print("   • Neural Network (256→128→64→32 layers)")
    print("   • Support Vector Machine (RBF kernel)")
    print("")
    print("🎲 CLUSTERING:")
    print("   • KMeans - 8 clusters")
    print("")
    print("📈 TOTAL: 8+ AI MODELS IN ENSEMBLE")
    print("=" * 80)
    
    while True:
        iteration += 1
        print(f"\n{'=' * 80}")
        print(f"🔄 ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 80}")
        
        success = False
        total_rounds = 0
        
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n📡 Attempt {attempt}/{MAX_RETRIES}")
            
            try:
                success, total_rounds = perform_scrape_and_predict()
                
                if success:
                    print(f"\n✅ Scrape successful! Total rounds: {total_rounds}")
                    consecutive_failures = 0
                    break
                else:
                    print(f"\n⚠️ Attempt {attempt} failed")
                    if attempt < MAX_RETRIES:
                        print(f"⏱️  Retrying in {RETRY_DELAY_SECONDS} seconds...")
                        time.sleep(RETRY_DELAY_SECONDS)
                        
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
        
        if not success:
            consecutive_failures += 1
            print(f"\n⚠️ Consecutive failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")
            
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n❌ Too many consecutive failures. Stopping.")
                break
        else:
            next_run = datetime.now() + timedelta(minutes=SCRAPE_INTERVAL_MINUTES)
            print(f"\n⏰ Next scrape: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💤 Sleeping for {SCRAPE_INTERVAL_MINUTES} minutes...")
            time.sleep(SCRAPE_INTERVAL_MINUTES * 60)

def main():
    """Main function"""
    try:
        run_scraping_loop()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Stopped by user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
