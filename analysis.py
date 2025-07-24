import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_database():
    # Veritabanına bağlanma
    conn = sqlite3.connect('sports_data.db')
    cursor = conn.cursor()
    
    print("=== VERİTABANI ANALİZİ ===\n")
    
    # Veri kalitesi kontrolü
    check_data_quality(conn)
    
    # Temel tablo analizleri
    analyze_table_structures(cursor)
    
    # Lig bazlı detaylı analiz
    analyze_leagues(conn)
    
    # Sezon bazlı analiz
    analyze_seasons(conn)
    
    # Ev sahibi/deplasman analizi
    analyze_home_away(conn)
    
    # Gol analizleri
    analyze_goals(conn)
    
    # Maç sonuçları analizi
    analyze_match_results(conn)
    
    conn.close()

def analyze_table_structures(cursor):
    """Tablo yapılarını analiz et"""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\n=== {table_name} TABLOSU ANALİZİ ===")
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print("\nTablo Yapısı:")
        for col in columns:
            print(f"- {col[1]} ({col[2]})")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"\nToplam Kayıt Sayısı: {count}")
        
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        print("\nÖrnek Veriler (İlk 5 kayıt):")
        for row in sample_data:
            print(row)

def analyze_leagues(conn):
    """Lig bazlı detaylı analiz"""
    print("\n=== LİG BAZLI DETAYLI ANALİZ ===")
    
    leagues_query = """
    SELECT 
        l.league_name,
        c.country_name,
        COUNT(DISTINCT f.fixture_key) as match_count,
        COUNT(DISTINCT t.team_key) as team_count,
        AVG(f.ft_home_score + f.ft_away_score) as avg_goals_per_match,
        AVG(f.ft_home_score) as avg_home_goals,
        AVG(f.ft_away_score) as avg_away_goals,
        SUM(CASE WHEN f.ft_home_score > f.ft_away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as home_win_percentage,
        SUM(CASE WHEN f.ft_home_score = f.ft_away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as draw_percentage,
        SUM(CASE WHEN f.ft_home_score < f.ft_away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as away_win_percentage
    FROM leagues l
    JOIN countries c ON l.country_key = c.country_key
    LEFT JOIN fixtures f ON l.league_key = f.league_key
    LEFT JOIN teams t ON l.league_key = t.league_key
    GROUP BY l.league_key
    ORDER BY match_count DESC
    """
    
    leagues_df = pd.read_sql_query(leagues_query, conn)
    print("\nLig İstatistikleri:")
    print(leagues_df.round(2))

def analyze_seasons(conn):
    """Sezon bazlı analiz"""
    print("\n=== SEZON BAZLI ANALİZ ===")
    
    seasons_query = """
    SELECT 
        league_season,
        COUNT(*) as match_count,
        AVG(ft_home_score + ft_away_score) as avg_goals_per_match,
        AVG(ht_home_score + ht_away_score) as avg_ht_goals_per_match,
        SUM(CASE WHEN ft_home_score > ft_away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as home_win_percentage,
        SUM(CASE WHEN ft_home_score = ft_away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as draw_percentage,
        SUM(CASE WHEN ft_home_score < ft_away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as away_win_percentage
    FROM fixtures
    GROUP BY league_season
    ORDER BY league_season DESC
    """
    
    seasons_df = pd.read_sql_query(seasons_query, conn)
    print("\nSezon İstatistikleri:")
    print(seasons_df.round(2))

def analyze_home_away(conn):
    """Ev sahibi/deplasman analizi"""
    print("\n=== EV SAHİBİ/DEPLASMAN ANALİZİ ===")
    
    home_away_query = """
    SELECT 
        l.league_name,
        AVG(f.ft_home_score) as avg_home_goals,
        AVG(f.ft_away_score) as avg_away_goals,
        SUM(CASE WHEN f.ft_home_score > f.ft_away_score THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as home_win_percentage,
        AVG(CASE WHEN f.ft_home_score > 0 AND f.ft_away_score > 0 THEN 1 ELSE 0 END) * 100 as both_teams_score_percentage,
        AVG(CASE WHEN (f.ft_home_score + f.ft_away_score) > 2.5 THEN 1 ELSE 0 END) * 100 as over_2_5_percentage
    FROM fixtures f
    JOIN leagues l ON f.league_key = l.league_key
    GROUP BY l.league_key
    ORDER BY home_win_percentage DESC
    """
    
    home_away_df = pd.read_sql_query(home_away_query, conn)
    print("\nEv Sahibi/Deplasman İstatistikleri:")
    print(home_away_df.round(2))

def analyze_goals(conn):
    """Gol analizleri"""
    print("\n=== GOL ANALİZLERİ ===")
    
    goals_query = """
    SELECT 
        l.league_name,
        AVG(f.ft_home_score + f.ft_away_score) as avg_total_goals,
        AVG(f.ht_home_score + f.ht_away_score) as avg_ht_goals,
        AVG((f.ft_home_score - f.ht_home_score) + (f.ft_away_score - f.ht_away_score)) as avg_sh_goals,
        SUM(CASE WHEN (f.ft_home_score + f.ft_away_score) > 2.5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as over_2_5_percentage,
        SUM(CASE WHEN f.ft_home_score > 0 AND f.ft_away_score > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as btts_percentage,
        AVG(CASE 
            WHEN (f.ht_home_score + f.ht_away_score) > ((f.ft_home_score - f.ht_home_score) + (f.ft_away_score - f.ht_away_score)) THEN 1
            WHEN (f.ht_home_score + f.ht_away_score) < ((f.ft_home_score - f.ht_home_score) + (f.ft_away_score - f.ht_away_score)) THEN 2
            ELSE 0 
        END) as most_goals_half
    FROM fixtures f
    JOIN leagues l ON f.league_key = l.league_key
    GROUP BY l.league_key
    ORDER BY avg_total_goals DESC
    """
    
    goals_df = pd.read_sql_query(goals_query, conn)
    print("\nGol İstatistikleri:")
    print(goals_df.round(2))

def analyze_match_results(conn):
    """Maç sonuçları analizi"""
    print("\n=== MAÇ SONUÇLARI ANALİZİ ===")
    
    results_query = """
    SELECT 
        l.league_name,
        COUNT(*) as total_matches,
        SUM(CASE WHEN ft_home_score > ft_away_score THEN 1 ELSE 0 END) as home_wins,
        SUM(CASE WHEN ft_home_score = ft_away_score THEN 1 ELSE 0 END) as draws,
        SUM(CASE WHEN ft_home_score < ft_away_score THEN 1 ELSE 0 END) as away_wins,
        AVG(CASE WHEN ft_home_score = ht_home_score AND ft_away_score = ht_away_score THEN 1 ELSE 0 END) * 100 as no_change_after_ht_percentage,
        AVG(CASE WHEN (ft_home_score > ft_away_score AND ht_home_score <= ht_away_score) OR 
                     (ft_home_score < ft_away_score AND ht_home_score >= ht_away_score) THEN 1 ELSE 0 END) * 100 as comeback_percentage
    FROM fixtures f
    JOIN leagues l ON f.league_key = l.league_key
    GROUP BY l.league_key
    ORDER BY comeback_percentage DESC
    """
    
    results_df = pd.read_sql_query(results_query, conn)
    print("\nMaç Sonuçları İstatistikleri:")
    print(results_df.round(2))

def check_data_quality(conn):
    """Veri kalitesi kontrolü"""
    print("\n=== VERİ KALİTESİ KONTROLÜ ===")
    
    # Eksik veri kontrolü
    missing_query = """
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN event_date IS NULL THEN 1 ELSE 0 END) as missing_date,
        SUM(CASE WHEN event_time IS NULL THEN 1 ELSE 0 END) as missing_time,
        SUM(CASE WHEN home_team_key IS NULL THEN 1 ELSE 0 END) as missing_home_team,
        SUM(CASE WHEN away_team_key IS NULL THEN 1 ELSE 0 END) as missing_away_team,
        SUM(CASE WHEN ht_home_score IS NULL THEN 1 ELSE 0 END) as missing_ht_home,
        SUM(CASE WHEN ht_away_score IS NULL THEN 1 ELSE 0 END) as missing_ht_away,
        SUM(CASE WHEN ft_home_score IS NULL THEN 1 ELSE 0 END) as missing_ft_home,
        SUM(CASE WHEN ft_away_score IS NULL THEN 1 ELSE 0 END) as missing_ft_away
    FROM fixtures
    """
    
    # Tutarsız veri kontrolü
    inconsistent_query = """
    SELECT 
        COUNT(*) as total_matches,
        SUM(CASE WHEN ft_home_score < ht_home_score THEN 1 ELSE 0 END) as invalid_home_score,
        SUM(CASE WHEN ft_away_score < ht_away_score THEN 1 ELSE 0 END) as invalid_away_score,
        SUM(CASE WHEN ft_home_score > 20 OR ft_away_score > 20 THEN 1 ELSE 0 END) as suspicious_high_score,
        SUM(CASE WHEN ABS(ft_home_score - ft_away_score) > 10 THEN 1 ELSE 0 END) as suspicious_goal_diff
    FROM fixtures
    """
    
    # Sezon geçiş kontrolü
    season_check_query = """
    SELECT 
        league_season,
        MIN(event_date) as season_start,
        MAX(event_date) as season_end,
        COUNT(*) as match_count
    FROM fixtures
    GROUP BY league_season
    ORDER BY season_start
    """
    
    missing_df = pd.read_sql_query(missing_query, conn)
    inconsistent_df = pd.read_sql_query(inconsistent_query, conn)
    season_df = pd.read_sql_query(season_check_query, conn)
    
    print("\nEksik Veri Analizi:")
    print(missing_df)
    
    print("\nTutarsız Veri Analizi:")
    print(inconsistent_df)
    
    print("\nSezon Geçiş Analizi:")
    print(season_df)

if __name__ == "__main__":
    analyze_database() 