import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib
from feature_engineering import FeatureEngineering

class ModelTraining:
    def __init__(self, db_path='sports_data.db'):
        self.fe = FeatureEngineering(db_path)
        self.models = {}
        self.scalers = {}
    
    def _convert_goal_diff_to_class(self, diff):
        """Gol farkını sınıf etiketine dönüştür"""
        if diff <= -3:
            return "-3 ve altı"
        elif diff >= 3:
            return "+3 ve üstü"
        else:
            return f"{diff:+d}"
    
    def prepare_training_data(self, league_key):
        """Belirli bir lig için eğitim verilerini hazırla"""
        # Lig maçlarını al
        df = self.fe.get_league_fixtures(league_key)
        
        X = []  # Özellikler
        y_ht_winner = []  # İlk yarı kazananı
        y_ft_winner = []  # Maç sonu kazananı
        y_ht_score = []  # İlk yarı skoru
        y_ft_score = []  # Maç sonu skoru
        y_ht_goals = []  # İlk yarı gol sayısı (1.5 alt/üst)
        y_ht_home = []   # İlk yarı ev sahibi gol sayısı (0.5 alt/üst)
        y_ht_away = []   # İlk yarı deplasman gol sayısı (0.5 alt/üst)
        y_ft_goals = []  # Maç sonu gol sayısı (2.5 alt/üst)
        y_ft_home = []   # Maç sonu ev sahibi gol sayısı (1.5 alt/üst)
        y_ft_away = []   # Maç sonu deplasman gol sayısı (1.5 alt/üst)
        y_ft_range = []  # Maç sonu gol aralığı (0-1/2-3/4-5/6+)
        y_both_score = [] # İlk yarı karşılıklı gol
        y_ft_both_score = [] # Maç sonu karşılıklı gol
        y_sh_both_score = [] # İkinci yarı karşılıklı gol
        y_most_goals_half = [] # En çok gol olan yarı
        
        # Yeni hedef değişkenler
        y_ht_home_15 = []  # İlk yarı ev gol sayısı (1.5 alt/üst)
        y_ht_away_15 = []  # İlk yarı deplasman gol sayısı (1.5 alt/üst)
        y_ht_goals_05 = [] # İlk yarı toplam gol sayısı (0.5 alt/üst)
        y_ht_goals_25 = [] # İlk yarı toplam gol sayısı (2.5 alt/üst)
        y_ht_goal_diff = [] # İlk yarı gol farkı
        
        y_ft_goals_15 = [] # Maç sonu toplam gol sayısı (1.5 alt/üst)
        y_ft_goals_35 = [] # Maç sonu toplam gol sayısı (3.5 alt/üst)
        y_ft_goal_diff = [] # Maç sonu gol farkı
        
        y_sh_home_05 = []  # İkinci yarı ev gol sayısı (0.5 alt/üst)
        y_sh_away_05 = []  # İkinci yarı deplasman gol sayısı (0.5 alt/üst)
        y_sh_home_15 = []  # İkinci yarı ev gol sayısı (1.5 alt/üst)
        y_sh_away_15 = []  # İkinci yarı deplasman gol sayısı (1.5 alt/üst)
        y_sh_goals_05 = [] # İkinci yarı toplam gol sayısı (0.5 alt/üst)
        y_sh_goals_15 = [] # İkinci yarı toplam gol sayısı (1.5 alt/üst)
        y_sh_goals_25 = [] # İkinci yarı toplam gol sayısı (2.5 alt/üst)
        y_sh_winner = []   # İkinci yarı kazanan taraf
        y_sh_goal_diff = [] # İkinci yarı gol farkı
        
        for idx, match in df.iterrows():
            # Sezon tarihlerini kontrol et
            season = match['league_season']
            season_start = datetime.strptime(f"{season.split('/')[0]}-07-01", '%Y-%m-%d')
            season_end = datetime.strptime(f"{season.split('/')[1]}-06-30", '%Y-%m-%d')
            match_date = datetime.strptime(match['event_date'], '%Y-%m-%d')
            
            if not (season_start <= match_date <= season_end):
                continue
                
            # Özellik çıkarımı
            features = self.fe.extract_features(
                league_key=match['league_key'],
                match_date=match['event_date'],
                match_time=match['event_time'],
                home_team_key=match['home_team_key'],
                away_team_key=match['away_team_key'],
                season=match['league_season']
            )
            
            # Özellik vektörünü oluştur
            feature_vector = [v for k, v in features.items() if k not in ['match_date', 'match_time', 'season']]
            X.append(feature_vector)
            
            # Hedef değişkenleri hazırla
            # İlk yarı kazananı (1=ev, 2=deplasman, 0=berabere)
            if match['ht_home_score'] > match['ht_away_score']:
                ht_winner = 1
            elif match['ht_home_score'] < match['ht_away_score']:
                ht_winner = 2
            else:
                ht_winner = 0
            
            # Maç sonu kazananı (1=ev, 2=deplasman, 0=berabere)
            if match['ft_home_score'] > match['ft_away_score']:
                ft_winner = 1
            elif match['ft_home_score'] < match['ft_away_score']:
                ft_winner = 2
            else:
                ft_winner = 0
            
            # Skorlar
            ht_score = f"{match['ht_home_score']}-{match['ht_away_score']}"
            ft_score = f"{match['ft_home_score']}-{match['ft_away_score']}"
            
            # Gol sayıları
            ht_total = match['ht_home_score'] + match['ht_away_score']
            ft_total = match['ft_home_score'] + match['ft_away_score']
            
            # Gol aralığı
            if ft_total <= 1:
                goal_range = '0-1'
            elif ft_total <= 3:
                goal_range = '2-3'
            elif ft_total <= 5:
                goal_range = '4-5'
            else:
                goal_range = '6+'
            
            # Karşılıklı gol
            both_score = 1 if match['ht_home_score'] > 0 and match['ht_away_score'] > 0 else 0
            ft_both_score = 1 if match['ft_home_score'] > 0 and match['ft_away_score'] > 0 else 0
            
            # İkinci yarı skorları
            sh_home_score = match['ft_home_score'] - match['ht_home_score']
            sh_away_score = match['ft_away_score'] - match['ht_away_score']
            sh_both_score = 1 if sh_home_score > 0 and sh_away_score > 0 else 0
            
            # İkinci yarı kazananı (1=ev, 2=deplasman, 0=berabere)
            if sh_home_score > sh_away_score:
                sh_winner = 1
            elif sh_home_score < sh_away_score:
                sh_winner = 2
            else:
                sh_winner = 0
            
            # En çok gol olan yarı (0=İlk Yarı, 1=İkinci Yarı, 2=Eşit)
            first_half_goals = match['ht_home_score'] + match['ht_away_score']
            second_half_goals = sh_home_score + sh_away_score
            if first_half_goals > second_half_goals:
                most_goals_half = 0  # İlk Yarı
            elif second_half_goals > first_half_goals:
                most_goals_half = 1  # İkinci Yarı
            else:
                most_goals_half = 2  # Eşit
            
            # Hedef değişkenleri ekle
            y_ht_winner.append(ht_winner)
            y_ft_winner.append(ft_winner)
            y_ht_score.append(ht_score)
            y_ft_score.append(ft_score)
            y_ht_goals.append(1 if ht_total > 1.5 else 0)
            y_ht_home.append(1 if match['ht_home_score'] > 0.5 else 0)
            y_ht_away.append(1 if match['ht_away_score'] > 0.5 else 0)
            y_ft_goals.append(1 if ft_total > 2.5 else 0)
            y_ft_home.append(1 if match['ft_home_score'] > 1.5 else 0)
            y_ft_away.append(1 if match['ft_away_score'] > 1.5 else 0)
            y_ft_range.append(goal_range)
            y_both_score.append(both_score)
            y_ft_both_score.append(ft_both_score)
            y_sh_both_score.append(sh_both_score)
            y_most_goals_half.append(most_goals_half)
            
            # Yeni hedef değişkenler
            y_ht_home_15.append(1 if match['ht_home_score'] > 1.5 else 0)
            y_ht_away_15.append(1 if match['ht_away_score'] > 1.5 else 0)
            y_ht_goals_05.append(1 if match['ht_home_score'] + match['ht_away_score'] > 0.5 else 0)
            y_ht_goals_25.append(1 if match['ht_home_score'] + match['ht_away_score'] > 2.5 else 0)
            y_ht_goal_diff.append(self._convert_goal_diff_to_class(match['ht_home_score'] - match['ht_away_score']))
            
            y_ft_goals_15.append(1 if ft_total > 1.5 else 0)
            y_ft_goals_35.append(1 if ft_total > 3.5 else 0)
            y_ft_goal_diff.append(self._convert_goal_diff_to_class(match['ft_home_score'] - match['ft_away_score']))
            
            y_sh_home_05.append(1 if sh_home_score > 0.5 else 0)
            y_sh_away_05.append(1 if sh_away_score > 0.5 else 0)
            y_sh_home_15.append(1 if sh_home_score > 1.5 else 0)
            y_sh_away_15.append(1 if sh_away_score > 1.5 else 0)
            y_sh_goals_05.append(1 if sh_home_score > 0.5 else 0)
            y_sh_goals_15.append(1 if sh_home_score > 1.5 else 0)
            y_sh_goals_25.append(1 if sh_home_score > 2.5 else 0)
            y_sh_winner.append(sh_winner)
            y_sh_goal_diff.append(self._convert_goal_diff_to_class(sh_home_score - sh_away_score))
        
        return np.array(X), {
            'ht_winner': np.array(y_ht_winner),
            'ft_winner': np.array(y_ft_winner),
            'ht_score': np.array(y_ht_score),
            'ft_score': np.array(y_ft_score),
            'ht_goals': np.array(y_ht_goals),
            'ht_home': np.array(y_ht_home),
            'ht_away': np.array(y_ht_away),
            'ft_goals': np.array(y_ft_goals),
            'ft_home': np.array(y_ft_home),
            'ft_away': np.array(y_ft_away),
            'ft_range': np.array(y_ft_range),
            'both_score': np.array(y_both_score),
            'ft_both_score': np.array(y_ft_both_score),
            'sh_both_score': np.array(y_sh_both_score),
            'most_goals_half': np.array(y_most_goals_half),
            'ht_home_15': np.array(y_ht_home_15),
            'ht_away_15': np.array(y_ht_away_15),
            'ht_goals_05': np.array(y_ht_goals_05),
            'ht_goals_25': np.array(y_ht_goals_25),
            'ht_goal_diff': np.array(y_ht_goal_diff),
            'ft_goals_15': np.array(y_ft_goals_15),
            'ft_goals_35': np.array(y_ft_goals_35),
            'ft_goal_diff': np.array(y_ft_goal_diff),
            'sh_home_05': np.array(y_sh_home_05),
            'sh_away_05': np.array(y_sh_away_05),
            'sh_home_15': np.array(y_sh_home_15),
            'sh_away_15': np.array(y_sh_away_15),
            'sh_goals_05': np.array(y_sh_goals_05),
            'sh_goals_15': np.array(y_sh_goals_15),
            'sh_goals_25': np.array(y_sh_goals_25),
            'sh_winner': np.array(y_sh_winner),
            'sh_goal_diff': np.array(y_sh_goal_diff)
        }
    
    def train_models(self, league_key):
        """Belirli bir lig için tüm modelleri eğit"""
        print(f"\nLig {league_key} için model eğitimi başlıyor...")
        
        # Veriyi hazırla
        X, y_dict = self.prepare_training_data(league_key)
        
        if len(X) == 0:
            print("Yeterli veri bulunamadı!")
            return
            
        if len(X) < 50:
            print("Yeterli veri yok! En az 50 maç gerekli.")
            return
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        self.models[league_key] = {}
        self.scalers[league_key] = {}
        
        for target in y_dict.keys():
            print(f"\n{target} için model eğitiliyor...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[league_key][target] = scaler
            
            if target in ['ht_goals', 'ht_home', 'ht_away', 
                         'ft_goals', 'ft_home', 'ft_away', 'both_score',
                         'ft_both_score', 'sh_both_score',
                         'ht_home_15', 'ht_away_15', 'ht_goals_05', 'ht_goals_25',
                         'ft_goals_15', 'ft_goals_35',
                         'sh_home_05', 'sh_away_05', 'sh_home_15', 'sh_away_15',
                         'sh_goals_05', 'sh_goals_15', 'sh_goals_25']:
                
                # RandomForest
                rf = RandomForestClassifier(
                    n_estimators=800,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    class_weight='balanced',
                    bootstrap=True,
                    criterion='gini',
                    warm_start=False,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                # XGBoost Classifier
                xgb_model = xgb.XGBClassifier(
                    n_estimators=1000,
                    max_depth=8,
                    learning_rate=0.02,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    tree_method='hist',
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    gamma=0.8,
                    scale_pos_weight=1,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0
                )
                
                # LightGBM Classifier
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=1000,
                    max_depth=10,
                    num_leaves=64,
                    learning_rate=0.02,
                    boosting_type='gbdt',
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_gain_to_split=0.01,
                    min_child_samples=30,
                    scale_pos_weight=1,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1
                )
                
                # Voting Ensemble
                ensemble = VotingClassifier(
                    estimators=[
                        ('rf', rf),
                        ('xgb', xgb_model),
                        ('lgb', lgb_model)
                    ],
                    voting='soft',  # Olasılık bazlı oylama
                    weights=[1,1,1],
                    n_jobs=-1
                )
                
                # Cross-validation
                cv_scores = cross_val_score(ensemble, X_scaled, y_dict[target], cv=kf, scoring='accuracy')
                print(f"Ensemble Cross-validation skorları: {cv_scores}")
                print(f"Ortalama CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                model = ensemble
                
            elif target in ['ht_winner', 'ft_winner','sh_winner','most_goals_half']:
                # Çoklu sınıf için benzer ensemble
                rf = RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=10,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    criterion='gini',
                    warm_start=False,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                # XGBoost Classifier
                xgb_model = xgb.XGBClassifier(
                    n_estimators=1000,
                    max_depth=8,
                    learning_rate=0.005,
                    min_child_weight=3,
                    tree_method='hist',
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.01,
                    reg_lambda=1.0,
                    gamma=1.0,
                    objective='multi:softprob',
                    num_class=3,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0
                )
                
                # LightGBM Classifier
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=1000,
                    max_depth=10,
                    num_leaves=64,
                    learning_rate=0.005,
                    min_child_samples=25,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.01,
                    reg_lambda=1.0,
                    min_split_gain=0.01,
                    objective='multiclass',
                    numclass=3,
                    boosting_type='gbdt',
                    bagging_fraction=0.8,
                    feature_fraction=0.8,
                    max_bin=255,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1
                )
                
                ensemble = VotingClassifier(
                    estimators=[
                        ('rf', rf),
                        ('xgb', xgb_model),
                        ('lgb', lgb_model)
                    ],
                    voting='soft',
                    weights=[1,1,1],
                    n_jobs=-1
                )
                
                cv_scores = cross_val_score(ensemble, X_scaled, y_dict[target], cv=kf, scoring='accuracy')
                print(f"Ensemble Cross-validation skorları: {cv_scores}")
                print(f"Ortalama CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                model = ensemble
            
            elif target in ['ht_score', 'ft_score']:
                # Çoklu sınıf için benzer ensemble
                rf = RandomForestClassifier(
                    n_estimators=1200,
                    max_depth=18,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    criterion='gini',
                    warm_start=False,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                # XGBoost Classifier
                xgb_model = xgb.XGBClassifier(
                    n_estimators=1200,
                    max_depth=10,
                    learning_rate=0.005,
                    min_child_weight=3,
                    tree_method='hist',
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.01,
                    reg_lambda=1.0,
                    gamma=2.0,
                    objective='multi:softprob',
                    num_class=64,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0
                )
                
                # LightGBM Classifier
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=1200,
                    max_depth=14,
                    num_leaves=128,
                    learning_rate=0.005,
                    min_child_samples=30,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.01,
                    reg_lambda=1.0,
                    min_split_gain=0.01,
                    objective='multiclass',
                    numclass=64,
                    boosting_type='gbdt',
                    bagging_fraction=0.85,
                    feature_fraction=0.85,
                    max_bin=255,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1
                )
                
                ensemble = VotingClassifier(
                    estimators=[
                        ('rf', rf),
                        ('xgb', xgb_model),
                        ('lgb', lgb_model)
                    ],
                    voting='soft',
                    weights=[1,1,1],
                    n_jobs=-1
                )
                
                cv_scores = cross_val_score(ensemble, X_scaled, y_dict[target], cv=kf, scoring='accuracy')
                print(f"Ensemble Cross-validation skorları: {cv_scores}")
                print(f"Ortalama CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                model = ensemble
            
            # Sonra regresyon bloğu
            elif target in ['ht_goal_diff', 'ft_goal_diff', 'sh_goal_diff', 'ft_range']:
                # Sınıflandırma modeli kullan
                rf = RandomForestClassifier(
                    n_estimators=800,
                    max_depth=10,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight='balanced',
                    bootstrap=True,
                    criterion='gini',
                    warm_start=False,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                # XGBoost Classifier
                xgb_model = xgb.XGBClassifier(
                    n_estimators=800,
                    max_depth=8,
                    learning_rate=0.01,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.001,
                    reg_lambda=1.0,
                    gamma=0.5,
                    objective='multi:softprob',
                    tree_method='hist',
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0
                )
                
                # LightGBM Classifier
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=800,
                    max_depth=10,
                    num_leaves=64,
                    learning_rate=0.01,
                    boosting_type='gbdt',
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.001,
                    reg_lambda=1.0,
                    min_split_gain=0.01,
                    objective='multiclass',
                    bagging_fraction=0.9,
                    feature_fraction=0.9,
                    max_bin=255,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1
                )
                
                # VotingClassifier ensemble
                ensemble = VotingClassifier(
                    estimators=[
                        ('rf', rf),
                        ('xgb', xgb_model),
                        ('lgb', lgb_model)
                    ],
                    voting='soft',
                    weights=[1,1,1],
                    n_jobs=-1
                )
                
                # Cross-validation
                cv_scores = cross_val_score(ensemble, X_scaled, y_dict[target], cv=kf, scoring='accuracy')
                print(f"Ensemble Cross-validation skorları: {cv_scores}")
                print(f"Ortalama CV skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                model = ensemble
            
            # Final model eğitimi
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_dict[target], test_size=0.2, random_state=42
            )
            
            model.fit(X_train, y_train)
            self.models[league_key][target] = model
            
            # Model değerlendirmesi
            y_pred = model.predict(X_test)
            
            if target in ['ht_winner', 'ft_winner', 'ht_goals', 'ht_home', 'ht_away',
                         'ft_goals', 'ft_home', 'ft_away', 'both_score',
                         'ft_both_score', 'sh_both_score', 'most_goals_half',
                         'ht_home_15', 'ht_away_15', 'ht_goals_05', 'ht_goals_25',
                         'ft_goals_15', 'ft_goals_35',
                         'sh_home_05', 'sh_away_05', 'sh_home_15', 'sh_away_15',
                         'sh_goals_05', 'sh_goals_15', 'sh_goals_25', 'sh_winner',
                         'ht_goal_diff', 'ft_goal_diff', 'sh_goal_diff']:
                accuracy = accuracy_score(y_test, y_pred)
                print(f"\nTest seti doğruluğu: {accuracy:.4f}")
                print("\nSınıflandırma Raporu:")
                print(classification_report(y_test, y_pred))
            else:
                accuracy = sum(y_pred == y_test) / len(y_pred)
                print(f"\nTest seti doğruluğu: {accuracy:.4f}")
        
        self.save_models(league_key)
    
    def save_models(self, league_key):
        """Eğitilmiş modelleri kaydet"""
        for target, model in self.models[league_key].items():
            model_path = f"models/league_{league_key}_{target}_model.joblib"
            scaler_path = f"models/league_{league_key}_{target}_scaler.joblib"
            joblib.dump(model, model_path)
            joblib.dump(self.scalers[league_key][target], scaler_path)
    
    def train_all_leagues(self):
        """Tüm ligler için modelleri eğit"""
        leagues = self.fe.get_leagues()
        for _, league in leagues.iterrows():
            self.train_models(league['league_key'])

if __name__ == "__main__":
    # Model eğitimini başlat
    mt = ModelTraining()
    #mt.train_all_leagues()
    mt.train_models(279)
    mt.train_models(288)
    mt.train_models(301)
    mt.train_models(302)
    mt.train_models(308)
    mt.train_models(319)
    mt.train_models(322)
# 49: "A-League Men",# 56: "Bundesliga",# 57: "Premyer Liqa",# 63: "First Division A",# 124: "HNL",# 134: "Czech Liga",# 135: "Superliga",# 152: "Premier League",# # 153: "Championship",# 154: "League One",# 164: "Ligue 2",# 168: "Ligue 1",# 171: "2. Bundesliga",# 175: "Bundesliga",# 178: "Super League 1",# 191: "NB I",# 206: "Serie B",# 207: "Serie A",# 244: "Eredivisie",# 245: "Eerste Divisie",# 259: "Ekstraklasa",# 266: "Primeira Liga",# 272: "Liga I",# 278: "Saudi League",# # 279: "Premiership",# 288: "Super Liga",# 301: "Segunda División",# 302: "La Liga",# 308: "Super League",# 319: "1. Lig",# 322: "Super Lig"
    
    
    
    
    