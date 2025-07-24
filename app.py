import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QDateEdit, QTimeEdit,
                           QPushButton, QGroupBox, QScrollArea, QFrame, QFormLayout)
from PyQt5.QtCore import Qt, QDate, QTime
from PyQt5.QtGui import QFont, QPalette, QColor
from prediction import MatchPredictor
from feature_engineering import FeatureEngineering

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Futbol Maç Tahmin Sistemi")
        self.setMinimumSize(800, 600)
        
        # Veritabanı bağlantısı ve tahmin sınıfları
        self.fe = FeatureEngineering()
        self.predictor = MatchPredictor()
        self.current_league = None
        
        # Font ayarları
        self.default_font = QFont("Verdana")
        self.default_font.setPointSize(8)
        self.default_font.setBold(True)
        self.default_font.setWeight(75)  # Normal bold'dan daha kalın
        
        self.title_font = QFont("Verdana")
        self.title_font.setPointSize(9)
        self.title_font.setBold(True)
        self.title_font.setWeight(75)  # Normal bold'dan daha kalın
        
        # Stil ayarları
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e272e;
            }
            QWidget {
                background-color: #1e272e;
                color: #ecf0f1;
                font-family: Verdana;
                font-weight: bold;
            }
            QGroupBox {
                border: 1px solid #34495e;
                border-radius: 3px;
                margin-top: 0.5em;
                padding-top: 8px;
                color: #ecf0f1;
                background-color: rgba(30, 39, 46, 0.85);
                background-image: url(logo.png);
                background-repeat: repeat-x repeat-y;
                background-position: center;
                font-family: Verdana;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0 3px 0 3px;
                color: #ecf0f1;
                background-color: rgba(30, 39, 46, 0.95);
                font-family: Verdana;
                font-weight: bold;
            }
            QComboBox, QDateEdit, QTimeEdit {
                background-color: #2c3e50;
                border: 1px solid #34495e;
                border-radius: 2px;
                padding: 3px;
                color: #ecf0f1;
                min-height: 20px;
                font-family: Verdana;
                font-weight: bold;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QComboBox:on {
                border: 1px solid #3498db;
            }
            QLabel {
                color: #ecf0f1;
                background-color: transparent;
                font-family: Verdana;
                font-weight: bold;
            }
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px;
                min-height: 25px;
                font-family: Verdana;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QScrollArea {
                background-color: #1e272e;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2c3e50;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #34495e;
                min-height: 15px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: #2c3e50;
            }
        """)
        
        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_widget.setLayout(main_layout)
        
        # Tahmin formu
        form_group = QGroupBox("Maç Bilgileri")
        form_group.setFont(self.title_font)
        form_layout = QFormLayout()
        form_layout.setSpacing(5)
        form_layout.setContentsMargins(10, 10, 10, 10)
        
        # Form elemanları
        self.league_combo = QComboBox()
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        self.date_edit = QDateEdit()
        self.time_edit = QTimeEdit()
        self.season_combo = QComboBox()
        
        
        # Form elemanlarını layouta ekle
        form_layout.addRow("Lig:", self.league_combo)
        form_layout.addRow("Ev Sahibi:", self.home_combo)
        form_layout.addRow("Deplasman:", self.away_combo)
        form_layout.addRow("Tarih:", self.date_edit)
        form_layout.addRow("Saat:", self.time_edit)
        form_layout.addRow("Sezon:", self.season_combo)
        
        
        form_group.setLayout(form_layout)
        main_layout.addWidget(form_group)
        
        # Tahmin butonu
        predict_button = QPushButton("Tahmin Yap")
        predict_button.setFont(self.title_font)
        predict_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        predict_button.clicked.connect(self.make_prediction)
        main_layout.addWidget(predict_button)
        
        # Sonuçlar için scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.results_layout = QVBoxLayout()
        scroll_widget.setLayout(self.results_layout)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # Tahmin sonuçları için label sözlükleri
        self.ht_labels = {}
        self.ft_labels = {}
        self.sh_labels = {}
        
        # İlk yarı tahmin sonuçları
        ht_group = QGroupBox("İlk Yarı Tahminleri")
        ht_group.setFont(self.title_font)
        ht_layout = QVBoxLayout()
        ht_layout.setSpacing(3)
        ht_layout.setContentsMargins(5, 5, 5, 5)
        
        self.ht_labels['winner'] = QLabel()
        self.ht_labels['score'] = QLabel()
        self.ht_labels['goals'] = QLabel()
        self.ht_labels['home'] = QLabel()
        self.ht_labels['away'] = QLabel()
        self.ht_labels['both_score'] = QLabel()
        self.ht_labels['home_15'] = QLabel()
        self.ht_labels['away_15'] = QLabel()
        self.ht_labels['goals_05'] = QLabel()
        self.ht_labels['goals_25'] = QLabel()
        self.ht_labels['goal_diff'] = QLabel()
        
        for label in self.ht_labels.values():
            label.setFont(self.default_font)
            ht_layout.addWidget(label)
        
        ht_group.setLayout(ht_layout)
        self.results_layout.addWidget(ht_group)
        
        # Maç sonu tahmin sonuçları
        ft_group = QGroupBox("Maç Sonu Tahminleri")
        ft_group.setFont(self.title_font)
        ft_layout = QVBoxLayout()
        ft_layout.setSpacing(3)
        ft_layout.setContentsMargins(5, 5, 5, 5)
        
        self.ft_labels['winner'] = QLabel()
        self.ft_labels['score'] = QLabel()
        self.ft_labels['goals'] = QLabel()
        self.ft_labels['home'] = QLabel()
        self.ft_labels['away'] = QLabel()
        self.ft_labels['both_score'] = QLabel()
        self.ft_labels['range'] = QLabel()
        self.ft_labels['goals_15'] = QLabel()
        self.ft_labels['goals_35'] = QLabel()
        self.ft_labels['goal_diff'] = QLabel()
        
        for label in self.ft_labels.values():
            label.setFont(self.default_font)
            ft_layout.addWidget(label)
        
        ft_group.setLayout(ft_layout)
        self.results_layout.addWidget(ft_group)
        
        # İkinci yarı tahmin sonuçları
        sh_group = QGroupBox("İkinci Yarı Tahminleri")
        sh_group.setFont(self.title_font)
        sh_layout = QVBoxLayout()
        sh_layout.setSpacing(3)
        sh_layout.setContentsMargins(5, 5, 5, 5)

        self.sh_labels['winner'] = QLabel()
        self.sh_labels['both_score'] = QLabel()
        self.sh_labels['most_goals_half'] = QLabel()
        self.sh_labels['home_05'] = QLabel()
        self.sh_labels['away_05'] = QLabel()
        self.sh_labels['home_15'] = QLabel()
        self.sh_labels['away_15'] = QLabel()
        self.sh_labels['goals_05'] = QLabel()
        self.sh_labels['goals_15'] = QLabel()
        self.sh_labels['goals_25'] = QLabel()
        self.sh_labels['goal_diff'] = QLabel()
        
        for label in self.sh_labels.values():
            label.setFont(self.default_font)
            sh_layout.addWidget(label)
        
        sh_group.setLayout(sh_layout)
        self.results_layout.addWidget(sh_group)
        
        # Form verilerini yükle
        self.setup_date_time()
        self.setup_season_round()
        self.load_leagues()
        
        # Sinyal bağlantıları
        self.league_combo.currentIndexChanged.connect(self.load_teams)
        
    def setup_date_time(self):
        """Tarih ve saat ayarlarını yap"""
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.time_edit.setTime(QTime.currentTime())
    
    def setup_season_round(self):
        """Sezon ve hafta bilgilerini ayarla"""
        seasons = ['2024/2025', '2023/2024', '2022/2023', '2021/2022', '2020/2021', '2019/2020']
        self.season_combo.addItems(seasons)
        
    
    def load_leagues(self):
        """Ligleri yükle"""
        try:
            leagues = pd.read_sql_query("""
                SELECT DISTINCT l.league_key, l.league_name, c.country_name
                FROM leagues l
                JOIN countries c ON l.country_key = c.country_key
                ORDER BY c.country_name, l.league_name
            """, self.fe.conn)
            
            for _, league in leagues.iterrows():
                self.league_combo.addItem(
                    f"{league['league_name']} ({league['country_name']})",
                    league['league_key']
                )
        except Exception as e:
            print(f"Error loading leagues: {str(e)}")
            error_label = QLabel("Ligler yüklenirken hata oluştu!")
            error_label.setStyleSheet("color: red;")
            error_label.setFont(self.default_font)
            self.results_layout.addWidget(error_label)
    
    def load_teams(self):
        """Seçili lige göre takımları yükle"""
        league_key = self.league_combo.currentData()
        if league_key is None:
            return
        
        # Veritabanından takımları al
        query = """
        SELECT team_key, team_name
        FROM teams
        WHERE league_key = ?
        ORDER BY team_name
        """
        
        teams = pd.read_sql_query(query, self.fe.conn, params=[league_key])
        
        # Combo boxları temizle
        self.home_combo.clear()
        self.away_combo.clear()
        
        # Takımları ekle
        for _, team in teams.iterrows():
            self.home_combo.addItem(team['team_name'], team['team_key'])
            self.away_combo.addItem(team['team_name'], team['team_key'])
    
    def clear_results(self):
        """Sonuç alanını temizle"""
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                layout = item.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                    
    def get_top_predictions(self, probabilities, top_n=15):
        """En yüksek olasılıklı tahminleri döndür"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        # Ana tahmini çıkar ve kalan en yüksek olasılıklı tahminleri döndür
        main_pred = sorted_probs[0][0]
        other_preds = [(pred, prob) for pred, prob in sorted_probs[1:] if pred != main_pred]
        return other_preds[:min(len(other_preds), top_n-1)]  # Ana tahmin hariç top_n-1 tahmin
    
    def add_result_group(self, title, predictions):
        """Sonuç grubu ekle"""
        group = QGroupBox(title)
        group.setFont(self.title_font)
        layout = QVBoxLayout()
        layout.setSpacing(3)
        layout.setContentsMargins(5, 5, 5, 5)
        
        for key, value in predictions.items():
            
                
            item_layout = QHBoxLayout()
            item_layout.setSpacing(3)
            
            # Etiket
            label = QLabel(f"{key}:")
            label.setFont(self.default_font)
            label.setMinimumWidth(150)
            item_layout.addWidget(label)
            
            # Değer
            if isinstance(value, dict):
                if 'Tahmin' in value:
                    if key in ['Skor', 'Gol Aralığı'] and 'Tüm Olasılıklar' in value:
                        # Ana tahmin
                        text_parts = [f"{value['Tahmin']} ({value['Olasılık']})"]
                        # Diğer tahminler (ana tahmin hariç)
                        other_preds = self.get_top_predictions(value['Tüm Olasılıklar'])
                        text_parts.extend([f"{pred} (%{prob*100:.1f})" for pred, prob in other_preds])
                        text = ", ".join(text_parts)
                    elif key == 'Gol Farkı' and 'Olasılıklar' in value:
                        # Ana tahmin
                        text_parts = [f"{value['Tahmin']} ({value['Olasılık']})"]
                        # Diğer olasılıklar (ana tahmin hariç)
                        tahmin = value['Tahmin']
                        olasılıklar = {k: v for k, v in value['Olasılıklar'].items() if k != tahmin}
                        text_parts.extend([f"{k} ({v})" for k, v in olasılıklar.items()])
                        text = ", ".join(text_parts)
                    elif 'Olasılıklar' in value:
                        # Ana tahmin
                        tahmin = value['Tahmin']
                        text_parts = [f"{tahmin} ({value['Olasılık']})"]
                        # Diğer olasılıklar (ana tahmin hariç)
                        olasılıklar = {k: v for k, v in value['Olasılıklar'].items() if k != tahmin}
                        text_parts.extend([f"{k} ({v})" for k, v in olasılıklar.items()])
                        text = ", ".join(text_parts)
                    else:
                        # Normal tahmin
                        text = f"{value['Tahmin']} ({value['Olasılık']})"
                    
                    value_label = QLabel(text)
                    value_label.setFont(self.default_font)  # Tahmin değerlerini de kalın yap
                    item_layout.addWidget(value_label)
            else:
                value_label = QLabel(str(value))
                value_label.setFont(self.default_font)  # Diğer değerleri de kalın yap
                item_layout.addWidget(value_label)
            
            item_layout.addStretch()
            layout.addLayout(item_layout)
        
        group.setLayout(layout)
        self.results_layout.addWidget(group)
    
    def make_prediction(self):
        """Tahmin yap ve sonuçları göster"""
        try:
            # Girdi değerlerini al
            league_key = self.league_combo.currentData()
            
            # Eğer lig değiştiyse modelleri yeniden yükle
            if self.current_league != league_key:
                self.predictor = MatchPredictor()  # Yeni bir predictor oluştur
                if not self.predictor.load_models(league_key):
                    raise Exception("Modeller yüklenemedi!")
                self.current_league = league_key
            
            home_team_key = self.home_combo.currentData()
            away_team_key = self.away_combo.currentData()
            match_date = self.date_edit.date().toString("yyyy-MM-dd")
            match_time = self.time_edit.time().toString("HH:mm")
            season = self.season_combo.currentText()
            
            
            print(f"\nMaking prediction for:")
            print(f"League: {league_key}")
            print(f"Home Team: {home_team_key}")
            print(f"Away Team: {away_team_key}")
            print(f"Date: {match_date}")
            print(f"Time: {match_time}")
            print(f"Season: {season}")
            
            
            # Tahmin yap
            predictions = self.predictor.predict_match(
                league_key=league_key,
                match_date=match_date,
                match_time=match_time,
                home_team_key=home_team_key,
                away_team_key=away_team_key,
                season=season,
                
            )
            
            if predictions:
                # Sonuçları göster
                self.clear_results()
                self.add_result_group("İlk Yarı Tahminleri", predictions['İlk Yarı Tahminleri'])
                self.add_result_group("İkinci Yarı Tahminleri", predictions['İkinci Yarı Tahminleri'])
                self.add_result_group("Maç Sonu Tahminleri", predictions['Maç Sonu Tahminleri'])
            else:
                # Hata durumunda
                self.clear_results()
                error_label = QLabel("Tahmin yapılamadı! Lütfen konsol çıktısını kontrol edin.")
                error_label.setStyleSheet("color: red;")
                error_label.setFont(self.default_font)
                self.results_layout.addWidget(error_label)
        except Exception as e:
            print(f"Error in make_prediction: {str(e)}")
            self.clear_results()
            error_label = QLabel(f"Hata: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_label.setFont(self.default_font)
            self.results_layout.addWidget(error_label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 