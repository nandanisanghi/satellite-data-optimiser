import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="Satellite Communication AI Optimization",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("🛰️ Satellite Communication AI Optimization")
st.markdown("""
This application provides a frontend for visualizing AI models that optimize satellite communication
bandwidth and signal transmission. Explore different models, simulate various conditions, and analyze
performance metrics to improve satellite communication efficiency.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Explorer", "Model Training", "Simulation", "Results & Predictions"]
)

# Initialize session state for storing trained models and results
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {}
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'show_loading' not in st.session_state:
    st.session_state.show_loading = False

# Function to generate synthetic satellite data (simplified version)
def generate_sample_data(num_satellites=5, time_period=24, sampling_rate=15, 
                        weather_conditions=None, include_anomalies=True):
    if weather_conditions is None:
        weather_conditions = ["Clear", "Cloudy", "Rain"]
    
    # Calculate number of data points
    num_samples = int((time_period * 60) / sampling_rate)
    
    # Create timestamp range
    start_time = datetime.now().replace(microsecond=0, second=0, minute=0)
    timestamps = [start_time + timedelta(minutes=i * sampling_rate) for i in range(num_samples)]
    
    # Initialize empty dataframe
    data = []
    
    # Generate data for each satellite
    for sat_id in range(1, num_satellites + 1):
        # Base characteristics for this satellite
        base_bandwidth = random.uniform(60, 90)  # Base bandwidth utilization (%)
        base_signal = random.uniform(-60, -40)   # Base signal strength (dB)
        base_latency = random.uniform(20, 100)   # Base latency (ms)
        base_throughput = random.uniform(80, 200)  # Base throughput (Mbps)
        base_error_rate = random.uniform(0.0001, 0.001)  # Base bit error rate
        
        # Orbit characteristics
        orbit_type = random.choice(["LEO", "MEO", "GEO"])
        if orbit_type == "LEO":
            altitude = random.uniform(500, 1500)  # Low Earth Orbit (km)
        elif orbit_type == "MEO":
            altitude = random.uniform(5000, 20000)  # Medium Earth Orbit (km)
        else:
            altitude = random.uniform(35000, 36000)  # Geostationary Orbit (km)
        
        for timestamp in timestamps:
            # Time-based factors
            hour = timestamp.hour
            
            # Time of day effect
            time_factor = 1.0
            if 8 <= hour < 12:  # Morning peak
                time_factor = 1.2
            elif 12 <= hour < 14:  # Lunch dip
                time_factor = 0.9
            elif 14 <= hour < 18:  # Afternoon peak
                time_factor = 1.3
            elif 18 <= hour < 22:  # Evening entertainment
                time_factor = 1.15
            elif 22 <= hour or hour < 6:  # Night low
                time_factor = 0.7
            
            # Weather effect
            weather = random.choice(weather_conditions)
            weather_factor = 1.0
            
            if weather == "Clear":
                weather_factor = 1.0
            elif weather == "Cloudy":
                weather_factor = 0.9
            elif weather == "Rain":
                weather_factor = 0.75
            elif weather == "Snow":
                weather_factor = 0.7
            elif weather == "Storm":
                weather_factor = 0.5
            
            # Calculate parameters with all factors applied
            bandwidth_util = min(100, base_bandwidth * time_factor * random.uniform(0.9, 1.1))
            signal_strength = base_signal * weather_factor * random.uniform(0.95, 1.05)
            latency = base_latency / weather_factor * random.uniform(0.92, 1.08)
            throughput = base_throughput * weather_factor * time_factor * random.uniform(0.9, 1.1)
            bit_error_rate = base_error_rate / weather_factor * random.uniform(0.8, 1.2)
            
            # Add anomalies occasionally if enabled
            if include_anomalies and random.random() < 0.02:  # 2% chance of anomaly
                anomaly_type = random.choice(["signal_drop", "latency_spike", "bandwidth_surge", "error_spike"])
                
                if anomaly_type == "signal_drop":
                    signal_strength *= random.uniform(0.4, 0.7)
                    throughput *= random.uniform(0.5, 0.8)
                elif anomaly_type == "latency_spike":
                    latency *= random.uniform(2.0, 5.0)
                elif anomaly_type == "bandwidth_surge":
                    bandwidth_util = min(100, bandwidth_util * random.uniform(1.2, 1.5))
                elif anomaly_type == "error_spike":
                    bit_error_rate *= random.uniform(10, 50)
            
            # Create data point
            data_point = {
                'timestamp': timestamp,
                'satellite_id': f"SAT-{sat_id:03d}",
                'orbit_type': orbit_type,
                'altitude': altitude,
                'weather': weather,
                'bandwidth_utilization': bandwidth_util,
                'signal_strength': signal_strength,
                'latency': latency,
                'throughput': throughput,
                'bit_error_rate': bit_error_rate,
                'time_factor': time_factor
            }
            
            data.append(data_point)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add some more derived features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
    df['signal_to_noise_ratio'] = -df['signal_strength'] / (df['bit_error_rate'] * 1000)
    df['packet_loss'] = df['bit_error_rate'] * 100 * random.uniform(0.8, 1.2)
    
    # Weather encoding
    weather_mapping = {
        'Clear': 0, 
        'Cloudy': 1, 
        'Rain': 2, 
        'Snow': 3, 
        'Storm': 4
    }
    df['weather_encoded'] = df['weather'].map(weather_mapping)
    
    # Calculate transmission efficiency metric
    df['transmission_efficiency'] = (df['throughput'] / (df['bandwidth_utilization'] + 0.1)) * (100 - df['latency'] / 10) / 100
    
    return df

# Function to plot bandwidth utilization
def plot_bandwidth_utilization(data):
    st.write("#### Bandwidth Utilization Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["By Satellite", "Over Time"])
    
    with tab1:
        # Calculate average bandwidth utilization by satellite
        if 'satellite_id' in data.columns:
            sat_bandwidth = data.groupby('satellite_id')['bandwidth_utilization'].agg(['mean', 'min', 'max']).reset_index()
            sat_bandwidth.columns = ['Satellite', 'Average Utilization', 'Min Utilization', 'Max Utilization']
            
            # Create a bar chart
            fig = px.bar(
                sat_bandwidth, 
                x='Satellite', 
                y='Average Utilization',
                error_y=sat_bandwidth['Max Utilization'] - sat_bandwidth['Average Utilization'],
                error_y_minus=sat_bandwidth['Average Utilization'] - sat_bandwidth['Min Utilization'],
                title='Average Bandwidth Utilization by Satellite',
                labels={'Average Utilization': 'Bandwidth Utilization (%)'},
                color='Average Utilization',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Plot bandwidth utilization over time
        if 'timestamp' in data.columns:
            # Create time-series plot
            # Group by timestamp if there are multiple satellites
            if 'satellite_id' in data.columns and data['satellite_id'].nunique() > 1:
                time_data = data.groupby('timestamp')['bandwidth_utilization'].mean().reset_index()
                title = 'Average Bandwidth Utilization Over Time (All Satellites)'
            else:
                time_data = data[['timestamp', 'bandwidth_utilization']]
                title = 'Bandwidth Utilization Over Time'
            
            fig = px.line(
                time_data, 
                x='timestamp', 
                y='bandwidth_utilization',
                title=title,
                labels={'bandwidth_utilization': 'Bandwidth Utilization (%)', 'timestamp': 'Time'},
                line_shape='spline'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Function to plot signal strength over time
def plot_signal_strength_time(data):
    st.write("#### Signal Strength Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Time Series", "By Weather", "Signal Quality"])
    
    with tab1:
        # Plot signal strength over time
        if 'timestamp' in data.columns:
            # Group by timestamp if there are multiple satellites
            if 'satellite_id' in data.columns and data['satellite_id'].nunique() > 1:
                # Let user select specific satellites or view average
                satellite_options = ['All Satellites (Average)'] + sorted(data['satellite_id'].unique().tolist())
                selected_satellite = st.selectbox("Select Satellite:", satellite_options)
                
                if selected_satellite == 'All Satellites (Average)':
                    time_data = data.groupby('timestamp')['signal_strength'].mean().reset_index()
                    title = 'Average Signal Strength Over Time (All Satellites)'
                else:
                    time_data = data[data['satellite_id'] == selected_satellite][['timestamp', 'signal_strength']]
                    title = f'Signal Strength Over Time: {selected_satellite}'
            else:
                time_data = data[['timestamp', 'signal_strength']]
                title = 'Signal Strength Over Time'
            
            fig = px.line(
                time_data, 
                x='timestamp', 
                y='signal_strength',
                title=title,
                labels={'signal_strength': 'Signal Strength (dB)', 'timestamp': 'Time'},
                line_shape='spline'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Analyze signal strength by weather condition
        if 'weather' in data.columns:
            weather_signal = data.groupby('weather')['signal_strength'].agg(['mean', 'min', 'max', 'count']).reset_index()
            weather_signal.columns = ['Weather', 'Average Signal', 'Min Signal', 'Max Signal', 'Count']
            
            # Sort by average signal strength
            weather_signal = weather_signal.sort_values('Average Signal')
            
            # Create a bar chart
            fig = px.bar(
                weather_signal, 
                x='Weather', 
                y='Average Signal',
                error_y=weather_signal['Max Signal'] - weather_signal['Average Signal'],
                error_y_minus=weather_signal['Average Signal'] - weather_signal['Min Signal'],
                title='Signal Strength by Weather Condition',
                labels={'Average Signal': 'Signal Strength (dB)'},
                color='Average Signal',
                color_continuous_scale='Viridis',
                text='Count'
            )
            
            fig.update_traces(texttemplate='%{text} samples', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Signal quality distribution
        # Categorize signal strength into quality levels
        signal_quality = []
        
        for signal in data['signal_strength']:
            if signal > -50:
                quality = "Excellent"
            elif signal > -70:
                quality = "Good"
            elif signal > -85:
                quality = "Fair"
            else:
                quality = "Poor"
            signal_quality.append(quality)
        
        quality_df = pd.DataFrame({
            'Signal Strength': data['signal_strength'],
            'Quality': signal_quality
        })
        
        # Create a histogram with color by quality
        fig = px.histogram(
            quality_df, 
            x='Signal Strength',
            color='Quality',
            category_orders={"Quality": ["Poor", "Fair", "Good", "Excellent"]},
            title='Signal Strength Distribution by Quality',
            color_discrete_map={
                'Poor': 'red',
                'Fair': 'orange',
                'Good': 'lightgreen',
                'Excellent': 'darkgreen'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to plot latency analysis
def plot_latency_analysis(data):
    st.write("#### Latency Analysis")
    
    # Create a line plot for latency
    if 'timestamp' in data.columns:
        if 'satellite_id' in data.columns and data['satellite_id'].nunique() > 1:
            time_data = data.groupby('timestamp')['latency'].mean().reset_index()
            title = 'Average Latency Over Time (All Satellites)'
        else:
            time_data = data[['timestamp', 'latency']]
            title = 'Latency Over Time'
        
        fig = px.line(
            time_data, 
            x='timestamp', 
            y='latency',
            title=title,
            labels={'latency': 'Latency (ms)', 'timestamp': 'Time'},
            line_shape='spline'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency distribution
    fig = px.histogram(
        data, 
        x='latency',
        nbins=30,
        title='Latency Distribution',
        labels={'latency': 'Latency (ms)'},
        color_discrete_sequence=['#0078D7']
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Function for simulation
def run_simulation(duration=12, num_satellites=3, weather="Clear Sky", traffic="Low Constant", 
                  orbit="Low Earth Orbit (LEO)", optimization_enabled=True):
    # Create a placeholder for simulation results
    baseline_data = []
    optimized_data = []
    
    # Set up simulation parameters
    sampling_interval = 5  # minutes
    num_samples = int((duration * 60) / sampling_interval)
    
    # Create time points
    start_time = datetime.now().replace(microsecond=0, second=0, minute=0)
    times = [start_time + timedelta(minutes=i * sampling_interval) for i in range(num_samples)]
    
    # Generate dummy satellite data
    for sat_id in range(1, num_satellites + 1):
        for t_idx, t in enumerate(times):
            # Get weather condition
            if weather == "Clear Sky":
                actual_weather = "Clear"
            elif weather == "Light Rain":
                actual_weather = random.choice(["Clear", "Cloudy", "Rain"])
            elif weather == "Heavy Rain":
                actual_weather = random.choice(["Cloudy", "Rain", "Rain", "Storm"])
            else:
                actual_weather = random.choice(["Clear", "Cloudy", "Rain"])
            
            # Get traffic level
            if traffic == "Low Constant":
                traffic_level = 0.3
            elif traffic == "High Constant":
                traffic_level = 0.8
            elif traffic == "Diurnal Cycle":
                hour = t.hour
                if hour < 6:  # Night low
                    traffic_level = 0.2
                elif hour < 10:  # Morning ramp
                    traffic_level = 0.4
                elif hour < 16:  # Daytime high
                    traffic_level = 0.7
                elif hour < 20:  # Evening peak
                    traffic_level = 0.9
                else:  # Late evening decline
                    traffic_level = 0.5
            else:
                traffic_level = random.uniform(0.2, 0.9)
            
            # Calculate baseline metrics
            bandwidth_util = min(100, 60 + traffic_level * 40) * random.uniform(0.95, 1.05)
            signal_strength = -50 * (1 if actual_weather == "Clear" else 0.9 if actual_weather == "Cloudy" else 0.7) * random.uniform(0.95, 1.05)
            latency = 50 * (1 if actual_weather == "Clear" else 1.2 if actual_weather == "Cloudy" else 1.5) * (1 + traffic_level * 0.5) * random.uniform(0.95, 1.05)
            throughput = 200 * (1 if actual_weather == "Clear" else 0.9 if actual_weather == "Cloudy" else 0.7) * (1 - traffic_level * 0.2) * random.uniform(0.95, 1.05)
            bit_error_rate = 0.0001 * (1 if actual_weather == "Clear" else 1.2 if actual_weather == "Cloudy" else 1.5) * (1 + traffic_level * 0.5) * random.uniform(0.9, 1.1)
            
            # Create data point
            baseline_point = {
                'timestamp': t,
                'satellite_id': f"SAT-{sat_id:03d}",
                'weather': actual_weather,
                'traffic_level': traffic_level,
                'bandwidth_utilization': bandwidth_util,
                'signal_strength': signal_strength,
                'latency': latency,
                'throughput': throughput,
                'bit_error_rate': bit_error_rate
            }
            
            baseline_data.append(baseline_point)
            
            # Create optimized data point (if optimization enabled)
            if optimization_enabled:
                opt_bandwidth = bandwidth_util * 0.8 * random.uniform(0.95, 1.05)
                opt_signal = signal_strength * 1.2 * random.uniform(0.95, 1.05)
                opt_latency = latency * 0.7 * random.uniform(0.95, 1.05)
                opt_throughput = throughput * 1.3 * random.uniform(0.95, 1.05)
                opt_error_rate = bit_error_rate * 0.6 * random.uniform(0.95, 1.05)
                
                optimized_point = {
                    'timestamp': t,
                    'satellite_id': f"SAT-{sat_id:03d}",
                    'weather': actual_weather,
                    'traffic_level': traffic_level,
                    'bandwidth_utilization': opt_bandwidth,
                    'signal_strength': opt_signal,
                    'latency': opt_latency,
                    'throughput': opt_throughput,
                    'bit_error_rate': opt_error_rate,
                    'optimization_applied': "AI Optimization"
                }
                
                optimized_data.append(optimized_point)
            else:
                optimized_data.append(baseline_point)
    
    # Create dataframes
    baseline_df = pd.DataFrame(baseline_data)
    optimized_df = pd.DataFrame(optimized_data)
    
    # Add improvement metrics
    if optimization_enabled:
        optimized_df['bandwidth_improvement'] = (baseline_df['bandwidth_utilization'] - optimized_df['bandwidth_utilization']) / baseline_df['bandwidth_utilization'] * 100
        optimized_df['signal_improvement'] = (optimized_df['signal_strength'] - baseline_df['signal_strength']) / abs(baseline_df['signal_strength']) * 100
        optimized_df['latency_improvement'] = (baseline_df['latency'] - optimized_df['latency']) / baseline_df['latency'] * 100
        optimized_df['throughput_improvement'] = (optimized_df['throughput'] - baseline_df['throughput']) / baseline_df['throughput'] * 100
        optimized_df['error_rate_improvement'] = (baseline_df['bit_error_rate'] - optimized_df['bit_error_rate']) / baseline_df['bit_error_rate'] * 100
        
        # Calculate overall optimization score
        optimized_df['optimization_score'] = (
            optimized_df['bandwidth_improvement'] * 0.2 +
            optimized_df['signal_improvement'] * 0.2 +
            optimized_df['latency_improvement'] * 0.2 +
            optimized_df['throughput_improvement'] * 0.2 +
            optimized_df['error_rate_improvement'] * 0.2
        )
    
    return baseline_df, optimized_df

# Function to plot simulation metrics comparison
def plot_simulation_metrics_comparison(baseline, optimized):
    st.write("#### Simulation Results: Performance Metrics Comparison")
    
    # Calculate average metrics for comparison
    baseline_metrics = baseline.mean()
    optimized_metrics = optimized.mean()
    
    # Create comparison dataframe
    metrics = ['bandwidth_utilization', 'signal_strength', 'latency', 'throughput', 'bit_error_rate']
    labels = ['Bandwidth Utilization (%)', 'Signal Strength (dB)', 'Latency (ms)', 'Throughput (Mbps)', 'Bit Error Rate']
    
    comparison_data = {
        'Metric': labels,
        'Baseline': [baseline_metrics[m] for m in metrics],
        'Optimized': [optimized_metrics[m] for m in metrics]
    }
    
    # Add improvement percentage
    comparison_data['Improvement (%)'] = [
        (comparison_data['Optimized'][i] - comparison_data['Baseline'][i]) / abs(comparison_data['Baseline'][i]) * 100
        if i != 0 and i != 2 and i != 4 else  # For signal and throughput, higher is better
        (comparison_data['Baseline'][i] - comparison_data['Optimized'][i]) / comparison_data['Baseline'][i] * 100
        for i in range(len(metrics))
    ]
    
    # Create a comparison table
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df.set_index('Metric').round(2))
    
    # Create a bar chart comparing baseline and optimized
    comparison_df_long = pd.melt(comparison_df, id_vars=['Metric'], value_vars=['Baseline', 'Optimized'],
                               var_name='Type', value_name='Value')
    
    # Plot for each metric
    for i, metric in enumerate(labels):
        metric_df = comparison_df_long[comparison_df_long['Metric'] == metric]
        
        fig = px.bar(
            metric_df,
            x='Metric',
            y='Value',
            color='Type',
            barmode='group',
            title=f'Comparison: {metric}',
            color_discrete_map={'Baseline': '#0078D7', 'Optimized': '#00B294'}
        )
        
        st.plotly_chart(fig, use_container_width=True)


# HOME PAGE
if app_mode == "Home":
    st.header("Satellite Communication Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About This Application")
        st.write("""
        The satellite communication industry faces several challenges:
        
        - **Bandwidth Optimization**: Limited bandwidth requires efficient allocation
        - **Signal Degradation**: Weather and environmental factors affect signal quality
        - **Latency Issues**: Particularly in deep-space communication
        - **Dynamic Resource Allocation**: Adapting to changing demand patterns
        
        This application provides tools to train AI models that can optimize these parameters,
        visualize communication data, and simulate different scenarios to improve performance.
        """)
        
        st.subheader("Getting Started")
        st.write("""
        1. Start with the **Data Explorer** to understand satellite communication parameters
        2. Use the **Model Training** section to train optimization algorithms
        3. Test your models in the **Simulation** environment
        4. Compare and analyze results in the **Results & Predictions** section
        """)
    
    with col2:
        st.subheader("Key Features")
        features = {
            "Data Visualization": "Interactive charts for bandwidth, signal strength, and more",
            "AI Model Training": "Train models to optimize communication parameters",
            "Simulation Environment": "Test algorithms in various scenarios",
            "Performance Metrics": "Compare and evaluate optimization results",
            "Prediction Capabilities": "Forecast signal degradation in adverse conditions"
        }
        
        for feature, description in features.items():
            st.markdown(f"**{feature}**: {description}")
        
        st.subheader("Common Metrics")
        metrics = {
            "Signal-to-Noise Ratio (SNR)": "Measures the quality of the signal compared to background noise",
            "Bandwidth Utilization": "Percentage of available bandwidth being effectively used",
            "Latency": "Time delay between transmission and reception",
            "Throughput": "Actual data transfer rate achieved",
            "Bit Error Rate (BER)": "Proportion of bits with errors relative to total bits transmitted"
        }
        
        for metric, description in metrics.items():
            st.markdown(f"**{metric}**: {description}")

# DATA EXPLORER PAGE
elif app_mode == "Data Explorer":
    st.header("Data Explorer")
    
    data_source = st.radio(
        "Select data source:",
        ["Generate sample data", "Upload your own data"]
    )
    
    if data_source == "Generate sample data":
        st.subheader("Generate Satellite Communication Data")
        
        with st.form("data_generation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                num_satellites = st.slider("Number of satellites", 1, 20, 5)
                time_period = st.slider("Time period (hours)", 1, 72, 24)
                sampling_rate = st.slider("Sampling rate (minutes)", 1, 60, 15)
                
            with col2:
                weather_conditions = st.multiselect(
                    "Weather conditions to include",
                    ["Clear", "Cloudy", "Rain", "Snow", "Storm"],
                    default=["Clear", "Cloudy", "Rain"]
                )
                
                include_anomalies = st.checkbox("Include random anomalies", value=True)
            
            generate_button = st.form_submit_button("Generate Data")
            
            if generate_button:
                with st.spinner("Generating satellite communication data..."):
                    st.session_state.show_loading = True
                    st.session_state.current_dataset = generate_sample_data(
                        num_satellites=num_satellites,
                        time_period=time_period,
                        sampling_rate=sampling_rate,
                        weather_conditions=weather_conditions,
                        include_anomalies=include_anomalies
                    )
                    
                    st.session_state.dataset_info = {
                        'satellites': num_satellites,
                        'time_period': time_period,
                        'sampling_rate': sampling_rate,
                        'weather_conditions': weather_conditions,
                        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.show_loading = False
                    st.success(f"Generated data for {num_satellites} satellites over {time_period} hours")
    
    elif data_source == "Upload your own data":
        st.subheader("Upload Satellite Communication Data")
        uploaded_file = st.file_uploader("Choose a CSV file with satellite data", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                required_columns = ["timestamp", "satellite_id", "bandwidth_utilization", 
                                   "signal_strength", "latency", "throughput", "bit_error_rate"]
                
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    st.error(f"The uploaded file is missing required columns: {', '.join(missing_columns)}")
                else:
                    st.session_state.current_dataset = data
                    st.session_state.dataset_info = {
                        'satellites': data['satellite_id'].nunique(),
                        'time_period': 24,  # Assuming 24 hours
                        'records': len(data),
                        'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.success("Data successfully loaded")
            except Exception as e:
                st.error(f"Error reading the uploaded file: {e}")
    
    # Display and visualize data if available
    if st.session_state.current_dataset is not None:
        data = st.session_state.current_dataset
        
        st.subheader("Dataset Overview")
        
        if st.session_state.dataset_info:
            info = st.session_state.dataset_info
            info_cols = st.columns(4)
            
            if 'satellites' in info:
                info_cols[0].metric("Satellites", info['satellites'])
            
            if 'time_period' in info:
                info_cols[1].metric("Time Period (hours)", round(info['time_period'], 1))
            
            if 'records' in info:
                info_cols[2].metric("Total Records", info['records'])
            else:
                info_cols[2].metric("Total Records", len(data))
            
            if 'weather_conditions' in info:
                info_cols[3].metric("Weather Conditions", len(info['weather_conditions']))
        
        st.write("### Data Sample")
        st.dataframe(data.head(10))
        
        st.write("### Data Statistics")
        st.dataframe(data.describe())
        
        st.subheader("Data Visualization")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Bandwidth Utilization", "Signal Strength Over Time", "Latency Analysis", 
             "Throughput vs Bandwidth", "Error Rate Analysis"]
        )
        
        # Create different visualizations based on selection
        if viz_type == "Bandwidth Utilization":
            plot_bandwidth_utilization(data)
        
        elif viz_type == "Signal Strength Over Time":
            plot_signal_strength_time(data)
        
        elif viz_type == "Latency Analysis":
            plot_latency_analysis(data)
        
        elif viz_type == "Throughput vs Bandwidth":
            st.write("#### Throughput vs Bandwidth Analysis")
            
            fig = px.scatter(
                data, 
                x='bandwidth_utilization', 
                y='throughput',
                color='satellite_id' if 'satellite_id' in data.columns else None,
                title='Throughput vs Bandwidth Utilization',
                labels={'bandwidth_utilization': 'Bandwidth Utilization (%)', 'throughput': 'Throughput (Mbps)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Error Rate Analysis":
            st.write("#### Bit Error Rate Analysis")
            
            # Plot error rate by weather
            if 'weather' in data.columns:
                weather_error = data.groupby('weather')['bit_error_rate'].mean().reset_index()
                
                fig = px.bar(
                    weather_error,
                    x='weather',
                    y='bit_error_rate',
                    title='Average Bit Error Rate by Weather Condition',
                    labels={'bit_error_rate': 'Bit Error Rate', 'weather': 'Weather Condition'},
                    color='bit_error_rate',
                    color_continuous_scale='Reds'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# MODEL TRAINING PAGE
elif app_mode == "Model Training":
    st.header("Model Training")
    
    if st.session_state.current_dataset is None:
        st.warning("No dataset available. Please generate or upload data in the Data Explorer section first.")
    else:
        data = st.session_state.current_dataset
        
        st.subheader("Configure and Train AI Models")
        st.info("Select parameters to train AI models for satellite communication optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Model Selection")
            model_type = st.selectbox(
                "Select model type:",
                ["Linear Regression", "Random Forest", "LSTM Neural Network", 
                 "Gradient Boosting", "Support Vector Regression"]
            )
            
            target_variable = st.selectbox(
                "Select optimization target:",
                ["bandwidth_utilization", "signal_strength", "latency", "throughput", "bit_error_rate"]
            )
            
            features = st.multiselect(
                "Select features to use:",
                [col for col in data.columns if col not in ['timestamp', target_variable, 'satellite_id', 'weather']],
                default=[col for col in data.columns if col not in ['timestamp', target_variable, 'satellite_id', 'weather']][:3]
            )
        
        with col2:
            st.write("### Training Parameters")
            
            test_size = st.slider("Test data percentage", 0.1, 0.5, 0.2, 0.05)
            
            if model_type == "Linear Regression":
                regularization = st.selectbox("Regularization type", ["None", "Ridge", "Lasso"])
                alpha = st.slider("Alpha (regularization strength)", 0.01, 1.0, 0.1)
                
            elif model_type == "Random Forest":
                n_estimators = st.slider("Number of trees", 10, 500, 100)
                max_depth = st.slider("Maximum depth", 2, 20, 10)
                
            elif model_type == "LSTM Neural Network":
                units = st.slider("LSTM Units", 32, 256, 128, 32)
                epochs = st.slider("Training Epochs", 10, 200, 50)
                batch_size = st.slider("Batch Size", 16, 128, 32, 8)
                
            elif model_type == "Gradient Boosting":
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                n_estimators_gb = st.slider("Number of estimators", 50, 500, 100)
                
            elif model_type == "Support Vector Regression":
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0, 0.1)
        
        model_name = st.text_input("Model name", f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if st.button("Train Model"):
            if not features:
                st.error("Please select at least one feature for training.")
            else:
                with st.spinner(f"Training {model_type} model..."):
                    # Set up training parameters
                    training_params = {
                        'model_type': model_type,
                        'target': target_variable,
                        'features': features,
                        'test_size': test_size
                    }
                    
                    # Add model-specific parameters
                    if model_type == "Linear Regression":
                        training_params.update({
                            'regularization': regularization,
                            'alpha': alpha
                        })
                    elif model_type == "Random Forest":
                        training_params.update({
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        })
                    elif model_type == "LSTM Neural Network":
                        training_params.update({
                            'units': units,
                            'epochs': epochs,
                            'batch_size': batch_size
                        })
                    elif model_type == "Gradient Boosting":
                        training_params.update({
                            'learning_rate': learning_rate,
                            'n_estimators': n_estimators_gb
                        })
                    elif model_type == "Support Vector Regression":
                        training_params.update({
                            'kernel': kernel,
                            'C': C
                        })
                    
                    # Simulate model training
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                    
                    # Store dummy model and results
                    metrics = {
                        'mean_absolute_error': round(random.uniform(0.05, 0.2), 3),
                        'mean_squared_error': round(random.uniform(0.01, 0.1), 3),
                        'root_mean_squared_error': round(random.uniform(0.1, 0.3), 3),
                        'r2_score': round(random.uniform(0.7, 0.95), 3),
                        'training_time': round(random.uniform(2, 10), 2)
                    }
                    
                    st.session_state.trained_models[model_name] = {
                        'params': training_params,
                        'metrics': metrics,
                        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success(f"Model '{model_name}' trained successfully!")
        
        # Display trained models if available
        if st.session_state.trained_models:
            st.subheader("Trained Models")
            
            for name, model_info in st.session_state.trained_models.items():
                with st.expander(f"Model: {name}"):
                    st.write(f"**Type:** {model_info['params']['model_type']}")
                    st.write(f"**Target:** {model_info['params']['target']}")
                    st.write(f"**Features:** {', '.join(model_info['params']['features'])}")
                    st.write(f"**Trained at:** {model_info['trained_at']}")
                    
                    st.write("#### Performance Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(model_info['metrics'].keys()),
                        'Value': list(model_info['metrics'].values())
                    })
                    st.table(metrics_df.set_index('Metric'))
                    
                    # Plot dummy feature importance
                    st.write("#### Feature Importance")
                    features = model_info['params']['features']
                    importance = np.random.random(len(features))
                    importance = importance / importance.sum()
                    
                    fig = px.bar(
                        x=features,
                        y=importance,
                        title='Feature Importance',
                        labels={'x': 'Feature', 'y': 'Importance'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# SIMULATION PAGE
elif app_mode == "Simulation":
    st.header("Simulation Environment")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train a model in the Model Training section first.")
    else:
        st.subheader("Configure Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select model to use
            model_names = list(st.session_state.trained_models.keys())
            selected_model = st.selectbox("Select trained model:", model_names)
            
            # Simulation parameters
            duration = st.slider("Simulation duration (hours)", 1, 48, 12)
            num_satellites = st.slider("Number of satellites", 1, 10, 3)
            
            weather = st.selectbox("Weather scenario:", 
                             ["Clear Sky", "Light Rain", "Heavy Rain", "Storm", "Variable Conditions"])
            
            traffic = st.selectbox("Traffic pattern:", 
                             ["Low Constant", "High Constant", "Diurnal Cycle", "Bursty", "Random"])
        
        with col2:
            orbit = st.selectbox("Orbit type:", 
                           ["Low Earth Orbit (LEO)", "Medium Earth Orbit (MEO)", "Geostationary Orbit (GEO)"])
            
            optimization_enabled = st.checkbox("Enable AI optimization", value=True)
            
            signal_degradation = st.slider("Signal degradation factor", 0.0, 0.5, 0.2, 0.05)
            bandwidth_constraint = st.slider("Bandwidth constraint factor", 0.5, 1.0, 0.8, 0.05)
            
            include_events = st.checkbox("Include random interference events", value=True)
            include_failures = st.checkbox("Include device failures", value=False)
        
        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                
                # Run simulation
                baseline_df, optimized_df = run_simulation(
                    duration=duration,
                    num_satellites=num_satellites,
                    weather=weather,
                    traffic=traffic,
                    orbit=orbit,
                    optimization_enabled=optimization_enabled
                )
                
                # Store simulation results
                simulation_id = f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.simulation_results[simulation_id] = {
                    'baseline': baseline_df,
                    'optimized': optimized_df,
                    'params': {
                        'model': selected_model,
                        'duration': duration,
                        'num_satellites': num_satellites,
                        'weather': weather,
                        'traffic': traffic,
                        'orbit': orbit,
                        'optimization_enabled': optimization_enabled
                    },
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.success(f"Simulation completed. Results stored with ID: {simulation_id}")
        
        # Display simulation results if available
        if st.session_state.simulation_results:
            st.subheader("Simulation Results")
            
            # Let user select which simulation to view
            sim_ids = list(st.session_state.simulation_results.keys())
            selected_sim = st.selectbox("Select simulation:", sim_ids, index=len(sim_ids)-1)
            
            sim_data = st.session_state.simulation_results[selected_sim]
            baseline = sim_data['baseline']
            optimized = sim_data['optimized']
            params = sim_data['params']
            
            # Display simulation parameters
            st.write("### Simulation Parameters")
            st.write(f"**Model:** {params['model']}")
            st.write(f"**Duration:** {params['duration']} hours")
            st.write(f"**Satellites:** {params['num_satellites']}")
            st.write(f"**Weather Scenario:** {params['weather']}")
            st.write(f"**Traffic Pattern:** {params['traffic']}")
            st.write(f"**Orbit Type:** {params['orbit']}")
            st.write(f"**AI Optimization:** {'Enabled' if params['optimization_enabled'] else 'Disabled'}")
            
            # Plot simulation results
            plot_simulation_metrics_comparison(baseline, optimized)
            
            # Plot time series comparison
            st.write("### Time Series Comparison")
            
            # Select metric to display
            metric = st.selectbox(
                "Select metric to compare:",
                ["bandwidth_utilization", "signal_strength", "latency", "throughput", "bit_error_rate"]
            )
            
            # Convert metric to display name
            metric_display = {
                "bandwidth_utilization": "Bandwidth Utilization (%)",
                "signal_strength": "Signal Strength (dB)",
                "latency": "Latency (ms)",
                "throughput": "Throughput (Mbps)",
                "bit_error_rate": "Bit Error Rate"
            }
            
            # Aggregate by time if multiple satellites
            if params['num_satellites'] > 1:
                baseline_time = baseline.groupby('timestamp')[metric].mean().reset_index()
                optimized_time = optimized.groupby('timestamp')[metric].mean().reset_index()
            else:
                baseline_time = baseline[['timestamp', metric]]
                optimized_time = optimized[['timestamp', metric]]
            
            # Create time series plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=baseline_time['timestamp'],
                y=baseline_time[metric],
                mode='lines',
                name='Baseline'
            ))
            
            fig.add_trace(go.Scatter(
                x=optimized_time['timestamp'],
                y=optimized_time[metric],
                mode='lines',
                name='Optimized'
            ))
            
            fig.update_layout(
                title=f"{metric_display[metric]} Comparison Over Time",
                xaxis_title="Time",
                yaxis_title=metric_display[metric]
            )
            
            st.plotly_chart(fig, use_container_width=True)

# RESULTS & PREDICTIONS PAGE
elif app_mode == "Results & Predictions":
    st.header("Results & Predictions")
    
    st.write("### Model Performance Comparison")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train models in the Model Training section first.")
    else:
        # Prepare data for model comparison
        models_data = []
        
        for name, model_info in st.session_state.trained_models.items():
            model_type = model_info['params']['model_type']
            target = model_info['params']['target']
            r2 = model_info['metrics']['r2_score']
            rmse = model_info['metrics']['root_mean_squared_error']
            
            models_data.append({
                'Model Name': name,
                'Model Type': model_type,
                'Target': target,
                'R² Score': r2,
                'RMSE': rmse
            })
        
        models_df = pd.DataFrame(models_data)
        
        # Create comparison plot
        fig = px.bar(
            models_df,
            x='Model Name',
            y='R² Score',
            color='Model Type',
            title='Model Performance Comparison (R² Score)',
            hover_data=['Target', 'RMSE']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Make predictions section
        st.write("### Make Predictions")
        
        # Select model for prediction
        model_names = list(st.session_state.trained_models.keys())
        selected_model = st.selectbox("Select model for prediction:", model_names)
        
        model_info = st.session_state.trained_models[selected_model]
        target = model_info['params']['target']
        
        st.write(f"Making predictions for: **{target}**")
        
        # Prediction inputs
        col1, col2 = st.columns(2)
        
        with col1:
            weather = st.selectbox("Weather condition:", ["Clear", "Cloudy", "Rain", "Snow", "Storm"])
            altitude = st.slider("Satellite altitude (km)", 500, 36000, 1000)
            time_of_day = st.selectbox("Time of day:", ["Morning", "Afternoon", "Evening", "Night"])
        
        with col2:
            traffic_level = st.slider("Traffic level", 0.0, 1.0, 0.5, 0.1)
            distance = st.slider("Distance to receiver (km)", 100, 5000, 1000)
        
        if st.button("Generate Prediction"):
            with st.spinner("Generating prediction..."):
                # Simulate prediction calculation
                time.sleep(2)
                
                # Generate a dummy prediction
                if target == 'bandwidth_utilization':
                    prediction = 75.0 + random.uniform(-10, 10)
                    result = {
                        'bandwidth_utilization': prediction,
                        'available_bandwidth': 100.0 - prediction,
                        'optimization_potential': min(20.0, (100.0 - prediction) * 0.5)
                    }
                elif target == 'signal_strength':
                    prediction = -55.0 + random.uniform(-10, 10)
                    quality = "Excellent" if prediction > -50 else "Good" if prediction > -70 else "Fair" if prediction > -85 else "Poor"
                    result = {
                        'signal_strength': prediction,
                        'signal_quality': quality,
                        'weather_impact': min(90, max(5, 30 if weather == "Storm" else 20 if weather == "Rain" else 10 if weather == "Cloudy" else 5))
                    }
                elif target == 'latency':
                    prediction = 50.0 + random.uniform(-10, 20)
                    result = {
                        'latency': prediction,
                        'network_congestion': min(95, max(5, traffic_level * 100)),
                        'distance_factor': max(1.0, distance / 1000)
                    }
                elif target == 'throughput':
                    prediction = 150.0 + random.uniform(-30, 30)
                    base_capacity = 10 + (altitude / 1000) * 5
                    weather_factor = 1.0 - (0.2 if weather == "Storm" else 0.1 if weather == "Rain" else 0.05 if weather == "Cloudy" else 0)
                    max_capacity = base_capacity * weather_factor
                    result = {
                        'throughput': prediction,
                        'max_capacity': max_capacity,
                        'utilization_rate': min(100, max(0, (prediction / max_capacity) * 100))
                    }
                else:  # bit_error_rate
                    prediction = 0.0005 + random.uniform(-0.0003, 0.0006)
                    interference = min(90, max(5, 40 if weather == "Storm" else 30 if weather == "Rain" else 20 if weather == "Cloudy" else 10))
                    result = {
                        'bit_error_rate': prediction,
                        'interference_level': interference,
                        'correction_capability': 95 if prediction < 0.0001 else 80 if prediction < 0.001 else 60 if prediction < 0.01 else 30
                    }
                
                # Display prediction results
                st.write("### Prediction Results")
                
                result_df = pd.DataFrame({
                    'Parameter': list(result.keys()),
                    'Value': list(result.values())
                })
                
                st.table(result_df.set_index('Parameter'))
                
                # Generate recommendations based on prediction
                st.write("### Recommendations")
                
                if target == 'bandwidth_utilization':
                    if result['bandwidth_utilization'] > 80:
                        st.info("**Implement Dynamic Bandwidth Allocation**: Current bandwidth utilization is high. Implement dynamic allocation algorithms to prioritize critical communications and defer non-essential transmissions.")
                    if traffic_level > 0.7:
                        st.info("**Schedule Non-Critical Transmissions**: High traffic detected. Consider scheduling non-critical data transmissions during off-peak hours.")
                    st.info("**Optimize Data Compression Algorithms**: Implement adaptive compression algorithms based on data type and priority to reduce transmission bandwidth requirements.")
                
                elif target == 'signal_strength':
                    if weather in ["Rain", "Snow", "Storm"]:
                        st.info(f"**Activate Weather Compensation Protocols**: Current {weather} conditions are degrading signal quality. Activate adaptive power and frequency adjustments to compensate for atmospheric interference.")
                    if result['signal_strength'] < -75:
                        st.info("**Increase Transmission Power**: Signal strength is below optimal levels. Consider temporarily increasing transmission power within regulatory limits.")
                    st.info("**Optimize Antenna Alignment**: Fine-tune ground station antenna alignment using predictive atmospheric models to compensate for current conditions.")
                
                elif target == 'latency':
                    st.info("**Implement Predictive Routing**: Use AI-based predictive routing to select optimal signal paths based on current network conditions and forecast changes.")
                    if distance > 2000:
                        st.info("**Deploy Edge Processing**: For long-distance communications, deploy edge processing to reduce round-trip data requirements.")
                    if traffic_level > 0.5:
                        st.info("**Implement Traffic Shaping**: Current network congestion is affecting latency. Implement QoS-based traffic shaping to prioritize time-sensitive communications.")
                
                elif target == 'throughput':
                    utilization = result['utilization_rate']
                    if utilization < 70:
                        st.info("**Optimize Modulation Schemes**: Current throughput is below potential. Implement adaptive modulation schemes based on current signal quality and interference levels.")
                    st.info("**Implement Channel Bonding**: Use multiple frequency channels simultaneously for high-priority transmissions to increase effective bandwidth.")
                
                elif target == 'bit_error_rate':
                    error_rate = result['bit_error_rate']
                    if error_rate > 0.001:
                        st.info("**Enhance Error Correction**: High bit error rate detected. Implement advanced error correction codes to improve data integrity.")
                    st.info("**Adaptive Coding and Modulation**: Implement ACM techniques to dynamically adjust modulation and coding based on current channel conditions.")