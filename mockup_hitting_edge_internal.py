import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
from PIL import Image
from streamlit_extras.app_logo import add_logo

# Mock data generation
def generate_mock_data(num_players=10, num_days=100):
    players = [f"Player {i}" for i in range(1, num_players + 1)]
    dates = pd.date_range(end=date.today(), periods=num_days)
    data = []
    
    for player in players:
        for d in dates:
            data.append({
                'player': player,
                'date': d,
                'bat_speed': np.random.normal(70, 5),
                'top_8th_ev': np.random.normal(95, 3),
                'expected_top_8th_ev': np.random.normal(97, 2),  # New metric
                'expected_top_8th_la': np.random.normal(25, 3),  # New metric
                'smash_factor': np.random.normal(1.4, 0.1),
                'attack_angle': np.random.normal(15, 5),
                'attack_angle_range': np.random.normal(5, 2),
                'point_of_contact': np.random.normal(0, 0.1),
                'swing_plus': np.random.normal(100, 10),
                'swing_decision': np.random.choice([0, 1]),  # 0 for bad, 1 for good
                'distance': np.random.normal(300, 50),
                'exit_velo': np.random.normal(90, 5),
                'level': np.random.choice(['Youth', 'High School', 'College', 'Professional']),
                'gym': np.random.choice(['WA', 'AZ', 'FL']),
                'trainer': np.random.choice(['Trainer A', 'Trainer B', 'Trainer C'])
            })
    
    return pd.DataFrame(data)

# Player Trends Page
def player_trends(df):
    st.header("Player Trends")
    
    # Sidebar
    player = st.sidebar.selectbox("Select Player", df['player'].unique())
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Filter data
    player_data = df[(df['player'] == player) & 
                     (df['date'].dt.date >= start_date) & 
                     (df['date'].dt.date <= end_date)]
    
    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Swing+", f"{player_data['swing_plus'].mean():.2f}")
    col2.metric("Bat Speed", f"{player_data['bat_speed'].mean():.2f}")
    col3.metric("Smash Factor", f"{player_data['smash_factor'].mean():.2f}")
    col4.metric("Top 8th EV", f"{player_data['top_8th_ev'].mean():.2f}")
    col5.metric("Avg Swing Decision", f"{player_data['swing_decision'].mean():.3f}")
    
    # New metrics
    col6, col7 = st.columns(2)
    col6.metric("Expected Top 8th EV", f"{player_data['expected_top_8th_ev'].mean():.2f}")
    col7.metric("Expected Top 8th LA", f"{player_data['expected_top_8th_la'].mean():.2f}")
    
    # Metric Graphs and Gains/Losses
    st.subheader("Metric Trends and Gains/Losses")
    metrics = ['bat_speed', 'top_8th_ev', 'expected_top_8th_ev', 'expected_top_8th_la', 'smash_factor', 'attack_angle', 'attack_angle_range', 'point_of_contact', 'swing_plus', 'swing_decision']
    selected_metrics = st.multiselect("Select metrics to display", metrics, default=['bat_speed', 'expected_top_8th_ev', 'expected_top_8th_la'])
    
    # Create two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Metric Graphs
        for metric in selected_metrics:
            fig = px.line(player_data, x='date', y=metric, title=f"{metric.replace('_', ' ').title()} Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gains/Losses (mock data)
        st.subheader("Gains/Losses")
        gains_losses = pd.DataFrame({
            'Metric': metrics,
            'Gain/Loss': np.random.uniform(-5, 5, len(metrics)),
            'Expected': np.random.uniform(-2, 2, len(metrics))
        })
        st.dataframe(gains_losses)
    
    # Data Table
    st.subheader("Data Table")
    show_data = st.checkbox("Show Data Table", value=False)
    if show_data:
        st.dataframe(player_data)
    else:
        st.info("Check the box above to display the data table.")

# In-Gym Trends Page
def in_gym_trends(df):
    st.header("In-Gym Trends")
    
    # Sidebar
    level = st.sidebar.multiselect("Select Level", ['Youth', 'High School', 'College', 'Professional'], default=['High School'])
    gym = st.sidebar.multiselect("Select Gym", ['WA', 'AZ', 'FL', 'Remote'], default=['WA', 'AZ', 'FL','Remote'])
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Filter data
    gym_data = df[(df['level'].isin(level)) & 
                  (df['gym'].isin(gym)) &
                  (df['date'].dt.date >= start_date) & 
                  (df['date'].dt.date <= end_date)]
    
    # Overall Gym Performance
    st.subheader("Overall Gym Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Bat Speed", f"{gym_data['bat_speed'].mean():.2f}")
    col2.metric("Avg Exit Velo", f"{gym_data['exit_velo'].mean():.2f}")
    col3.metric("Avg Smash Factor", f"{gym_data['smash_factor'].mean():.2f}")
    col4.metric("Avg Expected Top 8th EV", f"{gym_data['expected_top_8th_ev'].mean():.2f}")
    col5.metric("Avg Expected Top 8th LA", f"{gym_data['expected_top_8th_la'].mean():.2f}")
    
    # Leaderboards
    st.subheader("Leaderboards")
    leaderboard_metric = st.selectbox("Select Leaderboard", 
                                      ['distance', 'exit_velo', 'top_8th_ev', 'expected_top_8th_ev', 'expected_top_8th_la', 'bat_speed', 'smash_factor', 'swing_plus'])
    leaderboard = gym_data.groupby('player')[leaderboard_metric].mean().sort_values(ascending=False).head(10)
    st.bar_chart(leaderboard)
    
    # Trend Graphs
    st.subheader("Gym-wide Trends")
    metrics = ['bat_speed', 'exit_velo', 'smash_factor', 'swing_plus', 'expected_top_8th_ev', 'expected_top_8th_la']
    fig = go.Figure()
    for metric in metrics:
        trend = gym_data.groupby('date')[metric].mean()
        fig.add_trace(go.Scatter(x=trend.index, y=trend.values, mode='lines', name=metric))
    fig.update_layout(title="Gym-wide Trends Over Time", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)

    # Gym Comparison
    st.subheader("Gym Comparison")
    comparison_metric = st.selectbox("Select Metric for Gym Comparison", metrics)
    gym_comparison = gym_data.groupby('gym')[comparison_metric].mean().sort_values(ascending=False)
    st.bar_chart(gym_comparison)

# Trainer Trends Page
def trainer_trends(df):
    st.header("Trainer Performance Comparison")
    
    # Sidebar
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Level filter in sidebar
    use_level_filter = st.sidebar.checkbox("Filter by Athlete Level", value=False)
    if use_level_filter:
        levels = st.sidebar.multiselect("Select Athlete Levels", ['Youth', 'High School', 'College', 'Professional'], default=['High School', 'College'])
    else:
        levels = ['Youth', 'High School', 'College', 'Professional']
    
    # Filter data by date and level
    df_filtered = df[(df['date'].dt.date >= start_date) & 
                     (df['date'].dt.date <= end_date) & 
                     (df['level'].isin(levels))]
    
    # Individual Athlete Improvement (first section)
    st.subheader("Individual Athlete Improvement")
    selected_trainer = st.selectbox("Select Trainer", df_filtered['trainer'].unique())
    trainer_data = df_filtered[df_filtered['trainer'] == selected_trainer]
    
    metric = st.selectbox("Select metric", ['bat_speed', 'exit_velo', 'smash_factor', 'swing_plus', 'swing_decision', 'expected_top_8th_ev', 'expected_top_8th_la'])
    improvement = trainer_data.groupby(['player', 'date'])[metric].mean().unstack(level=0)
    fig = px.line(improvement, x=improvement.index, y=improvement.columns, title=f"Athlete {metric.replace('_', ' ').title()} Improvement")
    st.plotly_chart(fig)
    
    # Calculate overall averages
    overall_avg = df_filtered[['bat_speed', 'exit_velo', 'smash_factor', 'swing_plus', 'swing_decision', 'expected_top_8th_ev', 'expected_top_8th_la']].mean()
    
    # Calculate trainer averages
    trainer_avg = df_filtered.groupby('trainer')[['bat_speed', 'exit_velo', 'smash_factor', 'swing_plus', 'swing_decision', 'expected_top_8th_ev', 'expected_top_8th_la']].mean()
    
    # Calculate percentage difference from overall average
    trainer_performance = (trainer_avg - overall_avg) / overall_avg * 100
    
    # Visualize trainer performance (original view)
    st.subheader("Trainer Performance Visualization")
    
    fig = go.Figure()
    for trainer in trainer_performance.index:
        fig.add_trace(go.Bar(
            x=[trainer],
            y=[trainer_performance.loc[trainer, metric]],
            name=trainer
        ))
    
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Performance by Trainer",
        yaxis_title="% Difference from Average",
        xaxis_title="Trainer"
    )
    st.plotly_chart(fig)
    
    # Expandable section for Trainer Performance % and Best Performing Trainers
    with st.expander("**Trainer Performance Analysis**"):
        # Display trainer performance
        st.subheader("Trainer Performance (% difference from overall average)")
        st.dataframe(trainer_performance.style.format("{:.2f}%"))
        
        # Identify best performing trainer for each metric
        best_trainers = trainer_performance.idxmax()
        st.subheader("Best Performing Trainers")
        for metric, trainer in best_trainers.items():
            st.write(f"{metric.replace('_', ' ').title()}: {trainer} (+{trainer_performance.loc[trainer, metric]:.2f}%)")
    
    # Trainer Performance by Athlete Level
    st.subheader("Trainer Performance by Athlete Level")
    level_metric = st.selectbox("Select metric for level comparison", ['bat_speed', 'exit_velo', 'smash_factor', 'swing_plus', 'swing_decision', 'expected_top_8th_ev', 'expected_top_8th_la'], key='level_metric')
    trainer_level_performance = df_filtered.groupby(['trainer', 'level'])[level_metric].mean().unstack(level=1)
    st.bar_chart(trainer_level_performance)

# Main app
def main():
    st.set_page_config(page_title="Athlete KPI Dashboard", layout="wide")

    # Add the main logo using streamlit_extras
    add_logo("logo.png", height=65)

    # Add additional images to the sidebar
    st.sidebar.image("logo.png", caption="Chicks Dig the Long Ball ")
    
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.radio("Select Page", ["Player Trends", "In-Gym Trends", "Trainer Trends"])
    
    # Main content area title
    st.title("Athlete KPI Dashboard")
    
    # Generate mock data
    df = generate_mock_data()
    
    # Ensure 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    if page == "Player Trends":
        player_trends(df)
    elif page == "In-Gym Trends":
        in_gym_trends(df)
    elif page == "Trainer Trends":
        trainer_trends(df)

if __name__ == "__main__":
    main()
