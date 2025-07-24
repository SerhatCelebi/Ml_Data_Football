import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineering:
    def __init__(self, db_path='sports_data.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def get_leagues(self):
        """Tüm ligleri getir"""
        query = """
        SELECT DISTINCT l.league_key, l.league_name, c.country_name
        FROM leagues l
        JOIN countries c ON l.country_key = c.country_key
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_league_fixtures(self, league_key):
        """Belirli bir lig için tüm maçları getir"""
        query = """
        SELECT f.*, l.league_name, c.country_name
        FROM fixtures f
        JOIN leagues l ON f.league_key = l.league_key
        JOIN countries c ON l.country_key = c.country_key
        WHERE f.league_key = ?
        ORDER BY f.event_date, f.event_time
        """
        return pd.read_sql_query(query, self.conn, params=[league_key])
    
    def calculate_team_form(self, df, team_key, current_date, last_n_matches=6):
        """Takımın son n maçtaki form durumunu hesapla"""
        team_matches = df[
            ((df['home_team_key'] == team_key) | (df['away_team_key'] == team_key)) &
            (df['event_date'] < current_date)
        ].sort_values('event_date', ascending=False).head(last_n_matches)
        
        if len(team_matches) == 0:
            return {
                'win_rate': 0,
                'goals_scored_avg': 0,
                'goals_conceded_avg': 0,
                'clean_sheets': 0,
                'ft_both_score_rate': 0,
                'sh_both_score_rate': 0,
                'most_goals_half_first': 0,
                'most_goals_half_second': 0,
                'most_goals_half_equal': 0
            }
        
        wins = 0
        goals_scored = 0
        goals_conceded = 0
        clean_sheets = 0
        ft_both_score = 0
        sh_both_score = 0
        most_goals_half_first = 0
        most_goals_half_second = 0
        most_goals_half_equal = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team_key'] == team_key:
                goals_for = match['ft_home_score']
                goals_against = match['ft_away_score']
                if goals_for > goals_against:
                    wins += 1
                if goals_against == 0:
                    clean_sheets += 1
                
                # Karşılıklı gol kontrolü
                if goals_for > 0 and goals_against > 0:
                    ft_both_score += 1
                
                # İkinci yarı skorları
                sh_home_score = match['ft_home_score'] - match['ht_home_score']
                sh_away_score = match['ft_away_score'] - match['ht_away_score']
                if sh_home_score > 0 and sh_away_score > 0:
                    sh_both_score += 1
                
                # En çok gol olan yarı
                first_half_goals = match['ht_home_score'] + match['ht_away_score']
                second_half_goals = sh_home_score + sh_away_score
                if first_half_goals > second_half_goals:
                    most_goals_half_first += 1
                elif second_half_goals > first_half_goals:
                    most_goals_half_second += 1
                else:
                    most_goals_half_equal += 1
            else:
                goals_for = match['ft_away_score']
                goals_against = match['ft_home_score']
                if goals_for > goals_against:
                    wins += 1
                if goals_against == 0:
                    clean_sheets += 1
                
                # Karşılıklı gol kontrolü
                if goals_for > 0 and goals_against > 0:
                    ft_both_score += 1
                
                # İkinci yarı skorları
                sh_home_score = match['ft_home_score'] - match['ht_home_score']
                sh_away_score = match['ft_away_score'] - match['ht_away_score']
                if sh_home_score > 0 and sh_away_score > 0:
                    sh_both_score += 1
                
                # En çok gol olan yarı
                first_half_goals = match['ht_home_score'] + match['ht_away_score']
                second_half_goals = sh_home_score + sh_away_score
                if first_half_goals > second_half_goals:
                    most_goals_half_first += 1
                elif second_half_goals > first_half_goals:
                    most_goals_half_second += 1
                else:
                    most_goals_half_equal += 1
            
            goals_scored += goals_for
            goals_conceded += goals_against
        
        n_matches = len(team_matches)
        return {
            'win_rate': wins / n_matches,
            'goals_scored_avg': goals_scored / n_matches,
            'goals_conceded_avg': goals_conceded / n_matches,
            'clean_sheets': clean_sheets / n_matches,
            'ft_both_score_rate': ft_both_score / n_matches,
            'sh_both_score_rate': sh_both_score / n_matches,
            'most_goals_half_first': most_goals_half_first / n_matches,
            'most_goals_half_second': most_goals_half_second / n_matches,
            'most_goals_half_equal': most_goals_half_equal / n_matches
        }
    
    def calculate_h2h_stats(self, df, home_team_key, away_team_key, current_datetime, last_n_matches=6):
        """İki takım arasındaki son karşılaşmaların istatistiklerini hesapla"""
        h2h_matches = df[
            ((df['home_team_key'] == home_team_key) & (df['away_team_key'] == away_team_key) |
             (df['home_team_key'] == away_team_key) & (df['away_team_key'] == home_team_key)) &
            (df['event_date'] < current_datetime)
        ].sort_values('event_date', ascending=False).head(last_n_matches)
        
        if len(h2h_matches) == 0:
            return {
                'home_win_rate': 0,
                'away_win_rate': 0,
                'avg_goals': 0,
                'avg_ht_goals': 0,
                'ft_both_score_rate': 0,
                'sh_both_score_rate': 0,
                'most_goals_half_first': 0,
                'most_goals_half_second': 0,
                'most_goals_half_equal': 0
            }
        
        home_wins = 0
        away_wins = 0
        total_goals = 0
        total_ht_goals = 0
        ft_both_score = 0
        sh_both_score = 0
        most_goals_half_first = 0
        most_goals_half_second = 0
        most_goals_half_equal = 0
        
        for _, match in h2h_matches.iterrows():
            # Maç sonu ve ilk yarı toplam golleri
            total_goals += match['ft_home_score'] + match['ft_away_score']
            total_ht_goals += match['ht_home_score'] + match['ht_away_score']
            
            # Karşılıklı gol durumları
            if match['ft_home_score'] > 0 and match['ft_away_score'] > 0:
                ft_both_score += 1
            
            # İkinci yarı skorları
            sh_home_score = match['ft_home_score'] - match['ht_home_score']
            sh_away_score = match['ft_away_score'] - match['ht_away_score']
            if sh_home_score > 0 and sh_away_score > 0:
                sh_both_score += 1
            
            # En çok gol olan yarı
            first_half_goals = match['ht_home_score'] + match['ht_away_score']
            second_half_goals = sh_home_score + sh_away_score
            if first_half_goals > second_half_goals:
                most_goals_half_first += 1
            elif second_half_goals > first_half_goals:
                most_goals_half_second += 1
            else:
                most_goals_half_equal += 1
            
            # Kazanan takımı belirle
            if match['home_team_key'] == home_team_key:
                if match['ft_home_score'] > match['ft_away_score']:
                    home_wins += 1
                elif match['ft_home_score'] < match['ft_away_score']:
                    away_wins += 1
            else:
                if match['ft_home_score'] < match['ft_away_score']:
                    home_wins += 1
                elif match['ft_home_score'] > match['ft_away_score']:
                    away_wins += 1
        
        n_matches = len(h2h_matches)
        return {
            'home_win_rate': home_wins / n_matches,
            'away_win_rate': away_wins / n_matches,
            'avg_goals': total_goals / n_matches,
            'avg_ht_goals': total_ht_goals / n_matches,
            'ft_both_score_rate': ft_both_score / n_matches,
            'sh_both_score_rate': sh_both_score / n_matches,
            'most_goals_half_first': most_goals_half_first / n_matches,
            'most_goals_half_second': most_goals_half_second / n_matches,
            'most_goals_half_equal': most_goals_half_equal / n_matches
        }
    
    def calculate_season_stats(self, df, team_key, current_date, season):
        """Sezon içindeki takım istatistiklerini hesapla"""
        season_matches = df[
            ((df['home_team_key'] == team_key) | (df['away_team_key'] == team_key)) &
            (df['event_date'] < current_date) &
            (df['league_season'] == season)
        ]
        
        if len(season_matches) == 0:
            return {
                'season_win_rate': 0,
                'season_goals_per_game': 0,
                'season_conceded_per_game': 0,
                'season_ft_both_score_rate': 0,
                'season_sh_both_score_rate': 0,
                'season_most_goals_half_first': 0,
                'season_most_goals_half_second': 0,
                'season_most_goals_half_equal': 0
            }
        
        wins = 0
        goals_scored = 0
        goals_conceded = 0
        ft_both_score = 0
        sh_both_score = 0
        most_goals_half_first = 0
        most_goals_half_second = 0
        most_goals_half_equal = 0
        
        for _, match in season_matches.iterrows():
            if match['home_team_key'] == team_key:
                goals_scored += match['ft_home_score']
                goals_conceded += match['ft_away_score']
                if match['ft_home_score'] > match['ft_away_score']:
                    wins += 1
                
                # Karşılıklı gol kontrolü
                if match['ft_home_score'] > 0 and match['ft_away_score'] > 0:
                    ft_both_score += 1
                
                # İkinci yarı skorları
                sh_home_score = match['ft_home_score'] - match['ht_home_score']
                sh_away_score = match['ft_away_score'] - match['ht_away_score']
                if sh_home_score > 0 and sh_away_score > 0:
                    sh_both_score += 1
                
                # En çok gol olan yarı
                first_half_goals = match['ht_home_score'] + match['ht_away_score']
                second_half_goals = sh_home_score + sh_away_score
                if first_half_goals > second_half_goals:
                    most_goals_half_first += 1
                elif second_half_goals > first_half_goals:
                    most_goals_half_second += 1
                else:
                    most_goals_half_equal += 1
            else:
                goals_scored += match['ft_away_score']
                goals_conceded += match['ft_home_score']
                if match['ft_away_score'] > match['ft_home_score']:
                    wins += 1
                
                # Karşılıklı gol kontrolü
                if match['ft_home_score'] > 0 and match['ft_away_score'] > 0:
                    ft_both_score += 1
                
                # İkinci yarı skorları
                sh_home_score = match['ft_home_score'] - match['ht_home_score']
                sh_away_score = match['ft_away_score'] - match['ht_away_score']
                if sh_home_score > 0 and sh_away_score > 0:
                    sh_both_score += 1
                
                # En çok gol olan yarı
                first_half_goals = match['ht_home_score'] + match['ht_away_score']
                second_half_goals = sh_home_score + sh_away_score
                if first_half_goals > second_half_goals:
                    most_goals_half_first += 1
                elif second_half_goals > first_half_goals:
                    most_goals_half_second += 1
                else:
                    most_goals_half_equal += 1
        
        n_matches = len(season_matches)
        return {
            'season_win_rate': wins / n_matches,
            'season_goals_per_game': goals_scored / n_matches,
            'season_conceded_per_game': goals_conceded / n_matches,
            'season_ft_both_score_rate': ft_both_score / n_matches,
            'season_sh_both_score_rate': sh_both_score / n_matches,
            'season_most_goals_half_first': most_goals_half_first / n_matches,
            'season_most_goals_half_second': most_goals_half_second / n_matches,
            'season_most_goals_half_equal': most_goals_half_equal / n_matches
        }
    
    def calculate_home_away_advantage(self, df, team_key, current_date, last_n_matches=6):
        """Ev sahibi ve deplasman performans farklarını hesapla"""
        home_matches = df[
            (df['home_team_key'] == team_key) &
            (df['event_date'] < current_date)
        ].sort_values('event_date', ascending=False).head(last_n_matches)
        
        away_matches = df[
            (df['away_team_key'] == team_key) &
            (df['event_date'] < current_date)
        ].sort_values('event_date', ascending=False).head(last_n_matches)
        
        home_stats = {
            'goals_scored': 0,
            'goals_conceded': 0,
            'wins': 0,
            'matches': len(home_matches)
        }
        
        away_stats = {
            'goals_scored': 0,
            'goals_conceded': 0,
            'wins': 0,
            'matches': len(away_matches)
        }
        
        for _, match in home_matches.iterrows():
            home_stats['goals_scored'] += match['ft_home_score']
            home_stats['goals_conceded'] += match['ft_away_score']
            if match['ft_home_score'] > match['ft_away_score']:
                home_stats['wins'] += 1
        
        for _, match in away_matches.iterrows():
            away_stats['goals_scored'] += match['ft_away_score']
            away_stats['goals_conceded'] += match['ft_home_score']
            if match['ft_away_score'] > match['ft_home_score']:
                away_stats['wins'] += 1
        
        if home_stats['matches'] > 0:
            home_stats['goals_scored_avg'] = home_stats['goals_scored'] / home_stats['matches']
            home_stats['goals_conceded_avg'] = home_stats['goals_conceded'] / home_stats['matches']
            home_stats['win_rate'] = home_stats['wins'] / home_stats['matches']
        else:
            home_stats.update({'goals_scored_avg': 0, 'goals_conceded_avg': 0, 'win_rate': 0})
        
        if away_stats['matches'] > 0:
            away_stats['goals_scored_avg'] = away_stats['goals_scored'] / away_stats['matches']
            away_stats['goals_conceded_avg'] = away_stats['goals_conceded'] / away_stats['matches']
            away_stats['win_rate'] = away_stats['wins'] / away_stats['matches']
        else:
            away_stats.update({'goals_scored_avg': 0, 'goals_conceded_avg': 0, 'win_rate': 0})
        
        return {
            'home_away_goal_diff': home_stats['goals_scored_avg'] - away_stats['goals_scored_avg'],
            'home_away_conceded_diff': home_stats['goals_conceded_avg'] - away_stats['goals_conceded_avg'],
            'home_away_win_rate_diff': home_stats['win_rate'] - away_stats['win_rate']
        }
    
    def extract_features(self, league_key, match_date, match_time, home_team_key, away_team_key, season):
        """Belirli bir maç için tüm özellikleri çıkar"""
        df = self.get_league_fixtures(league_key)
        current_datetime = f"{match_date} {match_time}"
        
        # Takımların form durumları
        home_form = self.calculate_team_form(df, home_team_key, current_datetime)
        away_form = self.calculate_team_form(df, away_team_key, current_datetime)
        
        # H2H istatistikleri
        h2h_stats = self.calculate_h2h_stats(df, home_team_key, away_team_key, current_datetime)
        
        # Sezon istatistikleri
        home_season = self.calculate_season_stats(df, home_team_key, current_datetime, season)
        away_season = self.calculate_season_stats(df, away_team_key, current_datetime, season)
        
        # Ev/Deplasman avantajları
        home_advantage = self.calculate_home_away_advantage(df, home_team_key, current_datetime)
        away_advantage = self.calculate_home_away_advantage(df, away_team_key, current_datetime)
        
        # Mevsimsel etki (ay bazlı)
        month = datetime.strptime(match_date, '%Y-%m-%d').month
        
        features = {
            'league_key': league_key,
            'match_date': match_date,
            'match_time': match_time,
            'home_team_key': home_team_key,
            'away_team_key': away_team_key,
            'season': season,
            'month': month,
            
            # Form özellikleri
            'home_win_rate': home_form['win_rate'],
            'home_goals_scored_avg': home_form['goals_scored_avg'],
            'home_goals_conceded_avg': home_form['goals_conceded_avg'],
            'home_clean_sheets': home_form['clean_sheets'],
            'home_ft_both_score_rate': home_form['ft_both_score_rate'],
            'home_sh_both_score_rate': home_form['sh_both_score_rate'],
            'home_most_goals_half_first': home_form['most_goals_half_first'],
            'home_most_goals_half_second': home_form['most_goals_half_second'],
            'home_most_goals_half_equal': home_form['most_goals_half_equal'],
            'away_win_rate': away_form['win_rate'],
            'away_goals_scored_avg': away_form['goals_scored_avg'],
            'away_goals_conceded_avg': away_form['goals_conceded_avg'],
            'away_clean_sheets': away_form['clean_sheets'],
            'away_ft_both_score_rate': away_form['ft_both_score_rate'],
            'away_sh_both_score_rate': away_form['sh_both_score_rate'],
            'away_most_goals_half_first': away_form['most_goals_half_first'],
            'away_most_goals_half_second': away_form['most_goals_half_second'],
            'away_most_goals_half_equal': away_form['most_goals_half_equal'],
            
            # H2H özellikleri
            'h2h_home_win_rate': h2h_stats['home_win_rate'],
            'h2h_away_win_rate': h2h_stats['away_win_rate'],
            'h2h_avg_goals': h2h_stats['avg_goals'],
            'h2h_avg_ht_goals': h2h_stats['avg_ht_goals'],
            'h2h_ft_both_score_rate': h2h_stats['ft_both_score_rate'],
            'h2h_sh_both_score_rate': h2h_stats['sh_both_score_rate'],
            'h2h_most_goals_half_first': h2h_stats['most_goals_half_first'],
            'h2h_most_goals_half_second': h2h_stats['most_goals_half_second'],
            'h2h_most_goals_half_equal': h2h_stats['most_goals_half_equal'],
            
            # Sezon özellikleri
            'home_season_win_rate': home_season['season_win_rate'],
            'home_season_goals_per_game': home_season['season_goals_per_game'],
            'home_season_conceded_per_game': home_season['season_conceded_per_game'],
            'home_season_ft_both_score_rate': home_season['season_ft_both_score_rate'],
            'home_season_sh_both_score_rate': home_season['season_sh_both_score_rate'],
            'home_season_most_goals_half_first': home_season['season_most_goals_half_first'],
            'home_season_most_goals_half_second': home_season['season_most_goals_half_second'],
            'home_season_most_goals_half_equal': home_season['season_most_goals_half_equal'],
            'away_season_win_rate': away_season['season_win_rate'],
            'away_season_goals_per_game': away_season['season_goals_per_game'],
            'away_season_conceded_per_game': away_season['season_conceded_per_game'],
            'away_season_ft_both_score_rate': away_season['season_ft_both_score_rate'],
            'away_season_sh_both_score_rate': away_season['season_sh_both_score_rate'],
            'away_season_most_goals_half_first': away_season['season_most_goals_half_first'],
            'away_season_most_goals_half_second': away_season['season_most_goals_half_second'],
            'away_season_most_goals_half_equal': away_season['season_most_goals_half_equal'],
            
            # Ev/Deplasman avantaj özellikleri
            'home_home_away_goal_diff': home_advantage['home_away_goal_diff'],
            'home_home_away_conceded_diff': home_advantage['home_away_conceded_diff'],
            'home_home_away_win_rate_diff': home_advantage['home_away_win_rate_diff'],
            'away_home_away_goal_diff': away_advantage['home_away_goal_diff'],
            'away_home_away_conceded_diff': away_advantage['home_away_conceded_diff'],
            'away_home_away_win_rate_diff': away_advantage['home_away_win_rate_diff']
        }
        
        return features

if __name__ == "__main__":
    # Test kodu
    fe = FeatureEngineering()
    leagues = fe.get_leagues()
    print("Ligler:")
    print(leagues)
    
    # Örnek bir maç için özellik çıkarımı
    test_features = fe.extract_features(
        league_key=152,  # Premier League
        match_date='2023-12-01',
        match_time='20:00',
        home_team_key=72,  # Örnek takım
        away_team_key=73,  # Örnek takım
        season='2023/2024'
    )
    print("\nÖrnek Özellikler:")
    for key, value in test_features.items():
        print(f"{key}: {value}") 