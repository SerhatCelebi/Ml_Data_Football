import joblib
import numpy as np
from feature_engineering import FeatureEngineering

class MatchPredictor:
    def __init__(self, db_path='sports_data.db'):
        self.fe = FeatureEngineering(db_path)
        self.models = {}
        self.scalers = {}
    
    def load_models(self, league_key):
        """Belirli bir lig için eğitilmiş modelleri yükle"""
        targets = [
            'ht_winner', 'ft_winner', 'ht_score', 'ft_score',
            'ht_goals', 'ht_home', 'ht_away', 'ft_goals',
            'ft_home', 'ft_away', 'ft_range', 'both_score',
            'ft_both_score', 'sh_both_score', 'most_goals_half',
            # Yeni hedefler
            'ht_home_15', 'ht_away_15', 'ht_goals_05', 'ht_goals_25',
            'ht_goal_diff', 'ft_goals_15', 'ft_goals_35', 'ft_goal_diff',
            'sh_home_05', 'sh_away_05', 'sh_home_15', 'sh_away_15',
            'sh_goals_05', 'sh_goals_15', 'sh_goals_25', 'sh_winner',
            'sh_goal_diff'
        ]
        
        self.models[league_key] = {}
        self.scalers[league_key] = {}
        
        for target in targets:
            model_path = f"models/league_{league_key}_{target}_model.joblib"
            scaler_path = f"models/league_{league_key}_{target}_scaler.joblib"
            
            try:
                self.models[league_key][target] = joblib.load(model_path)
                self.scalers[league_key][target] = joblib.load(scaler_path)
            except:
                print(f"Model yüklenemedi: {model_path}")
                return False
        
        return True
    
    def predict_match(self, league_key, match_date, match_time, 
                     home_team_key, away_team_key, season):
        """Maç tahmini yap"""
        try:
            print("\nLoading models...")
            if not self.models:
                self.load_models(league_key)
                if not self.models:
                    print("Failed to load models!")
                    return None
            print("Models loaded successfully")
            
            print("\nExtracting features...")
            features_dict = self.fe.extract_features(
                league_key=league_key,
                match_date=match_date,
                match_time=match_time,
                home_team_key=home_team_key,
                away_team_key=away_team_key,
                season=season
            )
            if features_dict is None:
                print("Failed to extract features!")
                return None
                
            # Sözlük formatındaki özellikleri numpy array'e dönüştür
            feature_names = [k for k in features_dict.keys() if k not in ['match_date', 'match_time', 'season']]
            features = np.array([[features_dict[k] for k in feature_names]])
            print("Features extracted successfully")
            
            print("\nMaking predictions...")
            predictions = {}
            probabilities = {}
            
            for target in self.models[league_key].keys():
                try:
                    print(f"\nPredicting {target}...")
                    model = self.models[league_key][target]
                    scaler = self.scalers[league_key][target]
                    
                    # Scale features
                    scaled_features = scaler.transform(features)
                    
                    # Make prediction
                    if target in ['ht_goal_diff', 'ft_goal_diff', 'sh_goal_diff']:
                        # Artık sınıflandırma tahmini yapıyoruz
                        pred = model.predict(scaled_features)[0]
                        prob = model.predict_proba(scaled_features)[0]
                        predictions[target] = pred
                        probabilities[target] = prob
                        print(f"Prediction for {target}: {pred}")
                        print(f"Probabilities: {prob}")
                    else:
                        # Diğer tahminler için mevcut kod
                        pred = model.predict(scaled_features)[0]
                        prob = model.predict_proba(scaled_features)[0]
                        predictions[target] = pred
                        probabilities[target] = prob
                        print(f"Prediction for {target}: {pred}")
                        print(f"Probabilities: {prob}")
                except Exception as e:
                    print(f"Error predicting {target}: {str(e)}")
                    continue
            
            if not predictions:
                print("No predictions were made!")
                return None
                
            print("\nFormatting predictions...")
            formatted_predictions = self._format_predictions(predictions, probabilities, league_key, scaled_features)
            print("Predictions formatted successfully")
            
            return formatted_predictions
            
        except Exception as e:
            print(f"Error in predict_match: {str(e)}")
            return None
    
    def _format_predictions(self, predictions, probabilities, league_key, X_scaled):
        """Tahminleri okunabilir formata dönüştür"""
        
        def _most_goals_half_to_text(code):
            """En çok gol olan yarı kodunu metne çevir"""
            if code == 0:
                return "İlk Yarı"
            elif code == 1:
                return "İkinci Yarı"
            else:
                return "Eşit"
        
        result = {
            'İlk Yarı Tahminleri': {
                'Kazanan': {
                    'Tahmin': self._winner_to_text(predictions['ht_winner']),
                    'Olasılık': f"%{max(probabilities['ht_winner'])*100:.1f}",
                    'Olasılıklar': {
                        'Berabere': f"%{probabilities['ht_winner'][0]*100:.1f}",
                        'Ev Sahibi': f"%{probabilities['ht_winner'][1]*100:.1f}",
                        'Deplasman': f"%{probabilities['ht_winner'][2]*100:.1f}"
                    }
                },
                'Skor': {
                    'Tahmin': predictions['ht_score'],
                    'Olasılık': f"%{max(self.models[league_key]['ht_score'].predict_proba(X_scaled)[0])*100:.1f}",
                    'Tüm Olasılıklar': dict(zip(
                        self.models[league_key]['ht_score'].classes_,
                        self.models[league_key]['ht_score'].predict_proba(X_scaled)[0]
                    ))
                },
                'Toplam 0.5 Gol': {
                    'Tahmin': 'Üst' if predictions['ht_goals_05'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['ht_goals_05']) * 100, 1)}"
                },
                'Toplam Gol 1.5 Üst': {
                    'Tahmin': 'Üst' if predictions['ht_goals'] == 1 else 'Alt',
                    'Olasılık': f"%{max(probabilities['ht_goals'])*100:.1f}"
                },
                'Toplam 2.5 Gol': {
                    'Tahmin': 'Üst' if predictions['ht_goals_25'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['ht_goals_25']) * 100, 1)}"
                },
                'Ev Sahibi 0.5 Üst': {
                    'Tahmin': 'Üst' if predictions['ht_home'] == 1 else 'Alt',
                    'Olasılık': f"%{max(probabilities['ht_home'])*100:.1f}"
                },
                'Ev Sahibi 1.5 Gol': {
                    'Tahmin': 'Üst' if predictions['ht_home_15'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['ht_home_15']) * 100, 1)}"
                },
                'Deplasman 0.5 Üst': {
                    'Tahmin': 'Üst' if predictions['ht_away'] == 1 else 'Alt',
                    'Olasılık': f"%{max(probabilities['ht_away'])*100:.1f}"
                },
                'Deplasman 1.5 Gol': {
                    'Tahmin': 'Üst' if predictions['ht_away_15'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['ht_away_15']) * 100, 1)}"
                },
                'İlk Yarı Karşılıklı Gol': {
                    'Tahmin': 'Var' if predictions['both_score'] == 1 else 'Yok',
                    'Olasılık': f"%{max(probabilities['both_score'])*100:.1f}"
                },
                'Gol Farkı': {
                    'Tahmin': predictions['ht_goal_diff'],
                    'Olasılık': f"%{max(probabilities['ht_goal_diff'])*100:.1f}",
                    'Olasılıklar': {
                        k: f"%{v*100:.1f}" for k, v in zip(
                            self.models[league_key]['ht_goal_diff'].classes_,
                            probabilities['ht_goal_diff']
                        )
                    }
                }
            },
            'İkinci Yarı Tahminleri': {
                'Kazanan': {
                    'Tahmin': self._winner_to_text(predictions['sh_winner']),
                    'Olasılık': f"%{max(probabilities['sh_winner'])*100:.1f}",
                    'Olasılıklar': {
                        'Berabere': f"%{probabilities['sh_winner'][0]*100:.1f}",
                        'Ev Sahibi': f"%{probabilities['sh_winner'][1]*100:.1f}",
                        'Deplasman': f"%{probabilities['sh_winner'][2]*100:.1f}"
                    }
                },
                'Toplam 0.5 Gol': {
                    'Tahmin': 'Üst' if predictions['sh_goals_05'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['sh_goals_05']) * 100, 1)}"
                },
                'Toplam 1.5 Gol': {
                    'Tahmin': 'Üst' if predictions['sh_goals_15'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['sh_goals_15']) * 100, 1)}"
                },
                'Toplam 2.5 Gol': {
                    'Tahmin': 'Üst' if predictions['sh_goals_25'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['sh_goals_25']) * 100, 1)}"
                },
                'Ev Sahibi 0.5 Gol': {
                    'Tahmin': 'Üst' if predictions['sh_home_05'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['sh_home_05']) * 100, 1)}"
                },
                'Ev Sahibi 1.5 Gol': {
                    'Tahmin': 'Üst' if predictions['sh_home_15'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['sh_home_15']) * 100, 1)}"
                },
                'Deplasman 0.5 Gol': {
                    'Tahmin': 'Üst' if predictions['sh_away_05'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['sh_away_05']) * 100, 1)}"
                },
                'Deplasman 1.5 Gol': {
                    'Tahmin': 'Üst' if predictions['sh_away_15'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['sh_away_15']) * 100, 1)}"
                },
                'İkinci Yarı Karşılıklı Gol': {
                    'Tahmin': 'Var' if predictions['sh_both_score'] == 1 else 'Yok',
                    'Olasılık': f"%{max(probabilities['sh_both_score'])*100:.1f}"
                },
                'Gol Farkı': {
                    'Tahmin': predictions['sh_goal_diff'],
                    'Olasılık': f"%{max(probabilities['sh_goal_diff'])*100:.1f}",
                    'Olasılıklar': {
                        k: f"%{v*100:.1f}" for k, v in zip(
                            self.models[league_key]['sh_goal_diff'].classes_,
                            probabilities['sh_goal_diff']
                        )
                    }
                }
            },
            'Maç Sonu Tahminleri': {
                'Kazanan': {
                    'Tahmin': self._winner_to_text(predictions['ft_winner']),
                    'Olasılık': f"%{max(probabilities['ft_winner'])*100:.1f}",
                    'Olasılıklar': {
                        'Berabere': f"%{probabilities['ft_winner'][0]*100:.1f}",
                        'Ev Sahibi': f"%{probabilities['ft_winner'][1]*100:.1f}",
                        'Deplasman': f"%{probabilities['ft_winner'][2]*100:.1f}"
                    }
                },
                'Skor': {
                    'Tahmin': predictions['ft_score'],
                    'Olasılık': f"%{max(self.models[league_key]['ft_score'].predict_proba(X_scaled)[0])*100:.1f}",
                    'Tüm Olasılıklar': dict(zip(
                        self.models[league_key]['ft_score'].classes_,
                        self.models[league_key]['ft_score'].predict_proba(X_scaled)[0]
                    ))
                },
                'Toplam 1.5 Gol': {
                    'Tahmin': 'Üst' if predictions['ft_goals_15'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['ft_goals_15']) * 100, 1)}"
                },
                'Toplam Gol 2.5 Üst': {
                    'Tahmin': 'Üst' if predictions['ft_goals'] == 1 else 'Alt',
                    'Olasılık': f"%{max(probabilities['ft_goals'])*100:.1f}"
                },
                'Toplam 3.5 Gol': {
                    'Tahmin': 'Üst' if predictions['ft_goals_35'] == 1 else 'Alt',
                    'Olasılık': f"%{round(max(probabilities['ft_goals_35']) * 100, 1)}"
                },
                'Ev Sahibi 1.5 Üst': {
                    'Tahmin': 'Üst' if predictions['ft_home'] == 1 else 'Alt',
                    'Olasılık': f"%{max(probabilities['ft_home'])*100:.1f}"
                },
                'Deplasman 1.5 Üst': {
                    'Tahmin': 'Üst' if predictions['ft_away'] == 1 else 'Alt',
                    'Olasılık': f"%{max(probabilities['ft_away'])*100:.1f}"
                },
                'Maç Sonu Karşılıklı Gol': {
                    'Tahmin': 'Var' if predictions['ft_both_score'] == 1 else 'Yok',
                    'Olasılık': f"%{max(probabilities['ft_both_score'])*100:.1f}"
                },
                'Gol Aralığı': {
                    'Tahmin': predictions['ft_range'],
                    'Olasılık': f"%{max(self.models[league_key]['ft_range'].predict_proba(X_scaled)[0])*100:.1f}",
                    'Tüm Olasılıklar': dict(zip(
                        self.models[league_key]['ft_range'].classes_,
                        self.models[league_key]['ft_range'].predict_proba(X_scaled)[0]
                    ))
                },
                'En Çok Gol Olan Yarı': {
                    'Tahmin': _most_goals_half_to_text(predictions['most_goals_half']),
                    'Olasılık': f"%{max(probabilities['most_goals_half'])*100:.1f}",
                    'Olasılıklar': {
                        'İlk Yarı': f"%{probabilities['most_goals_half'][0]*100:.1f}",
                        'İkinci Yarı': f"%{probabilities['most_goals_half'][1]*100:.1f}",
                        'Eşit': f"%{probabilities['most_goals_half'][2]*100:.1f}"
                    }
                },
                'Gol Farkı': {
                    'Tahmin': predictions['ft_goal_diff'],
                    'Olasılık': f"%{max(probabilities['ft_goal_diff'])*100:.1f}",
                    'Olasılıklar': {
                        k: f"%{v*100:.1f}" for k, v in zip(
                            self.models[league_key]['ft_goal_diff'].classes_,
                            probabilities['ft_goal_diff']
                        )
                    }
                }
            }
        }
        
        return result
    
    def _winner_to_text(self, winner_code):
        """Kazanan kodunu metne çevir"""
        if winner_code == 0:
            return "Berabere"
        elif winner_code == 1:
            return "Ev Sahibi"
        else:
            return "Deplasman"

if __name__ == "__main__":
    # Test kodu
    predictor = MatchPredictor()
    
    # Örnek bir maç için tahmin yap
    predictions = predictor.predict_match(
        league_key=152,  # Premier League
        match_date='2024-01-20',
        match_time='16:00',
        home_team_key=72,  # Örnek takım
        away_team_key=73,  # Örnek takım
        season='2023/2024'
        
    )
    
    if predictions:
        print("\nTahmin Sonuçları:")
        print("\nİlk Yarı Tahminleri:")
        for key, value in predictions['İlk Yarı Tahminleri'].items():
            print(f"{key}: {value}")
        
        print("\nMaç Sonu Tahminleri:")
        for key, value in predictions['Maç Sonu Tahminleri'].items():
            print(f"{key}: {value}")
    else:
        print("Tahmin yapılamadı!") 