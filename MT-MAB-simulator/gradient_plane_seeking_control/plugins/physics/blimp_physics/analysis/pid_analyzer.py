import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration - Use relative paths to logs directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "pid_tuning_debug.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

def create_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

def load_data():
    """Load and return the PID tuning data"""
    try:
        abs_log_path = os.path.abspath(LOG_FILE_PATH)
        print(f"Looking for log file at: {abs_log_path}")
        
        if not os.path.exists(abs_log_path):
            print(f"Error: Log file not found at {abs_log_path}")
            print("Make sure to run the simulation with DEBUG=1 to generate log data")
            return None
            
        df = pd.read_csv(abs_log_path)
        print(f"Loaded {len(df)} data points")
        print(f"Controllers found: {df['controller'].unique()}")
        print(f"Time range: {df['time'].min():.3f}s to {df['time'].max():.3f}s")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_controller_response(df, controller_name, title):
    """Plot desired vs current for a specific controller"""
    controller_data = df[df['controller'] == controller_name].copy()
    
    if controller_data.empty:
        print(f"No data found for controller: {controller_name}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title} Controller Analysis', fontsize=16, fontweight='bold')
    
    time = controller_data['time']
    
    # Plot 1: Desired vs Current
    axes[0, 0].plot(time, controller_data['desired'], 'b-', label='Desired', linewidth=2)
    axes[0, 0].plot(time, controller_data['current'], 'r-', label='Current', linewidth=2)
    axes[0, 0].set_title('Desired vs Current Response')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error over time
    axes[0, 1].plot(time, controller_data['error'], 'g-', linewidth=2)
    axes[0, 1].set_title('Error Over Time')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 3: Control Output
    axes[1, 0].plot(time, controller_data['output'], 'm-', linewidth=2)
    axes[1, 0].set_title('Control Output')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Output')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: PID Components
    kp, ki, kd = get_pid_gains(controller_name)
    axes[1, 1].plot(time, controller_data['error'] * kp, label='P Component', alpha=0.7)
    axes[1, 1].plot(time, controller_data['integral'] * ki, label='I Component', alpha=0.7)
    axes[1, 1].plot(time, controller_data['derivative'] * kd, label='D Component', alpha=0.7)
    axes[1, 1].set_title('PID Components')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Contribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{controller_name.replace('_', '-')}-analysis.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    
    # Show plot
    plt.show()

def get_pid_gains(controller_name):
    """Get PID gains for controller (from your code)"""
    gains = {
        'speed': (100.0, 40.0, 100.0),
        'yaw_angle': (5.0, 0.0, 20.0),
        'yaw_rate': (120.0, 7.0, 0.0),
        'altitude_pos': (3.0, 0.005, 3.0),
        'altitude_vel': (55.0, 7.0, 1.0)
    }
    return gains.get(controller_name, (1.0, 0.0, 0.0))

def plot_performance_metrics(df):
    """Plot overall performance metrics"""
    controllers = df['controller'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PID Controller Performance Metrics', fontsize=16, fontweight='bold')
    
    # Calculate metrics for each controller
    metrics = {}
    for controller in controllers:
        data = df[df['controller'] == controller]
        if len(data) > 1:
            metrics[controller] = {
                'rms_error': np.sqrt(np.mean(data['error']**2)),
                'max_error': np.max(np.abs(data['error'])),
                'settling_time': calculate_settling_time(data),
                'steady_state_error': np.mean(data['error'].tail(50)) if len(data) > 50 else np.mean(data['error'])
            }
    
    # Plot metrics
    if metrics:
        controllers_list = list(metrics.keys())
        
        # RMS Error
        rms_errors = [metrics[c]['rms_error'] for c in controllers_list]
        axes[0, 0].bar(controllers_list, rms_errors, color='skyblue')
        axes[0, 0].set_title('RMS Error')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Max Error
        max_errors = [metrics[c]['max_error'] for c in controllers_list]
        axes[0, 1].bar(controllers_list, max_errors, color='lightcoral')
        axes[0, 1].set_title('Maximum Error')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Settling Time
        settling_times = [metrics[c]['settling_time'] for c in controllers_list]
        axes[1, 0].bar(controllers_list, settling_times, color='lightgreen')
        axes[1, 0].set_title('Settling Time (s)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Steady State Error
        ss_errors = [metrics[c]['steady_state_error'] for c in controllers_list]
        axes[1, 1].bar(controllers_list, ss_errors, color='gold')
        axes[1, 1].set_title('Steady State Error')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Print metrics to console
        print("\nPerformance Metrics:")
        print("-" * 50)
        for controller in controllers_list:
            m = metrics[controller]
            print(f"{controller:15} | RMS: {m['rms_error']:.4f} | Max: {m['max_error']:.4f} | "
                  f"Settling: {m['settling_time']:.2f}s | SS Error: {m['steady_state_error']:.4f}")
    
    plt.tight_layout()
    
    # Save plot
    filepath = os.path.join(OUTPUT_DIR, "performance-metrics.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    
    plt.show()

def calculate_settling_time(data, tolerance=0.02):
    """Calculate settling time (time to stay within tolerance)"""
    if len(data) < 10:
        return 0
    
    # Find steady state value (average of last 20% of data)
    steady_state = np.mean(data['current'].tail(int(len(data) * 0.2)))
    
    # Find when response stays within tolerance
    for i in range(len(data) - 20, 0, -1):
        if abs(data['current'].iloc[i] - steady_state) > tolerance * abs(steady_state):
            return data['time'].iloc[i]
    
    return data['time'].iloc[0]

def plot_physics_data():
    """Plot physics debug data if available"""
    physics_log_path = os.path.join(LOGS_DIR, "blimp_physics_debug.csv")
    
    if not os.path.exists(physics_log_path):
        print("Physics debug data not found - skipping physics plots")
        return
    
    try:
        df_physics = pd.read_csv(physics_log_path)
        print(f"\nLoaded {len(df_physics)} physics data points")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Physics Debug Data Analysis', fontsize=16, fontweight='bold')
        
        time = df_physics['time']
        
        # Plot 1: Raw control inputs
        axes[0, 0].plot(time, df_physics['thrust_raw'], label='Thrust', linewidth=2)
        axes[0, 0].plot(time, df_physics['pitch_raw'], label='Pitch', linewidth=2)
        axes[0, 0].plot(time, df_physics['yaw_raw'], label='Yaw', linewidth=2)
        axes[0, 0].set_title('Raw Control Inputs')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Control Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Individual propeller thrusts
        axes[0, 1].plot(time, df_physics['prop0'], label='Prop 0', linewidth=2)
        axes[0, 1].plot(time, df_physics['prop1'], label='Prop 1', linewidth=2)
        axes[0, 1].plot(time, df_physics['prop2'], label='Prop 2', linewidth=2)
        axes[0, 1].plot(time, df_physics['prop3'], label='Prop 3', linewidth=2)
        axes[0, 1].set_title('Individual Propeller Thrusts')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Thrust (N)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Total thrust vs differential thrust
        total_thrust = df_physics['prop0'] + df_physics['prop1'] + df_physics['prop2'] + df_physics['prop3']
        diff_thrust = (df_physics['prop2'] + df_physics['prop3']) - (df_physics['prop0'] + df_physics['prop1'])
        
        axes[1, 0].plot(time, total_thrust, 'b-', label='Total Thrust', linewidth=2)
        axes[1, 0].set_title('Total Thrust')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Total Thrust (N)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Thrust differential (for yaw control)
        axes[1, 1].plot(time, diff_thrust, 'r-', label='Thrust Differential', linewidth=2)
        axes[1, 1].set_title('Thrust Differential (Yaw Control)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Differential Thrust (N)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(OUTPUT_DIR, "physics-analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved physics plot: {filepath}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting physics data: {e}")

def main():
    """Main function to run all visualizations"""
    print("PID Controller Tuning Visualizer")
    print("=" * 40)
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Logs directory: {os.path.abspath(LOGS_DIR)}")
    
    # Create output directory
    create_output_dir()
    
    # Load PID data
    df = load_data()
    if df is None:
        print("\nNo PID data available. To generate data:")
        print("1. Build with: make DEBUG=1")
        print("2. Run your simulation")
        print("3. Run this script again")
        return
    
    # Plot each controller separately
    controller_titles = {
        'speed': 'Speed (Forward/Backward)',
        'yaw_angle': 'Yaw Angle',
        'yaw_rate': 'Yaw Rate',
        'altitude_pos': 'Altitude Position',
        'altitude_vel': 'Altitude Velocity'
    }
    
    available_controllers = df['controller'].unique()
    
    for controller, title in controller_titles.items():
        if controller in available_controllers:
            print(f"\nPlotting {title} controller...")
            plot_controller_response(df, controller, title)
        else:
            print(f"No data for {title} controller")
    
    # Plot performance metrics
    print("\nPlotting performance metrics...")
    plot_performance_metrics(df)
    
    # Plot physics data if available
    print("\nChecking for physics debug data...")
    plot_physics_data()
    
    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
