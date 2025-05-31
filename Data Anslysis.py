import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Path to your CSV files 
folder_path = r"C:/CARLA/CARLA_0.9.144"

# Extract speeds automatically from filenames
all_files = os.listdir(folder_path)
pid_files = sorted([f for f in all_files if f.startswith("pid_data") and f.endswith(".csv")])
mpc_files = sorted([f for f in all_files if f.startswith("mpc_data") and f.endswith(".csv")])

# Extract simulation times
def get_final_time(filepath):
    try:
        df = pd.read_csv(filepath)
        return df.iloc[-1, 0]  # last row, first column (time)
    except:
        return None

def extract_speed_from_name(name):
    return int(name.split("_")[-1].split(".")[0])

pid_times = []
mpc_times = []
pid_speeds = []
mpc_speeds = []

# Read PID times
for fname in pid_files:
    fullpath = os.path.join(folder_path, fname)
    time = get_final_time(fullpath)
    speed = extract_speed_from_name(fname)
    if time is not None:
        pid_speeds.append(speed)
        pid_times.append(time)

# Read MPC times
for fname in mpc_files:
    fullpath = os.path.join(folder_path, fname)
    time = get_final_time(fullpath)
    speed = extract_speed_from_name(fname)
    if time is not None:
        mpc_speeds.append(speed)
        mpc_times.append(time)

# Sort both by speed just in case
pid_sorted = sorted(zip(pid_speeds, pid_times))
mpc_sorted = sorted(zip(mpc_speeds, mpc_times))

pid_speeds, pid_times = zip(*pid_sorted)
mpc_speeds, mpc_times = zip(*mpc_sorted)

# lotting
plt.figure(figsize=(10, 6))
plt.plot(pid_speeds, pid_times, marker='o', label='PID', linewidth=2)
plt.plot(mpc_speeds, mpc_times, marker='s', label='MPC', linewidth=2)
plt.xlabel("Target Speed [km/h]")
plt.ylabel("Total Simulation Time [s]")
plt.title("Total Simulation Time vs Target Speed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def get_last_time(filepath):
    try:
        df = pd.read_csv(filepath)
        return df["time"].iloc[-1]
    except:
        return float("inf")

all_files = pid_files + mpc_files
min_duration = min(get_last_time(os.path.join(folder_path, f)) for f in all_files)

# Define clipped common time grid
dt = 0.1  # sampling interval
time_grid = np.arange(0, min_duration, dt)

def interpolate_cte(filepath, time_grid):
    try:
        df = pd.read_csv(filepath)
        time_vals = df["time"].values
        cte_vals = df["cte"].values
        # Clip interpolation strictly to valid data range (no extrapolation)
        interp_fn = interp1d(time_vals, cte_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        return interp_fn(time_grid)
    except:
        return None

pid_cte_all = []
mpc_cte_all = []

for fname in pid_files:
    cte_interp = interpolate_cte(os.path.join(folder_path, fname), time_grid)
    if cte_interp is not None:
        pid_cte_all.append(cte_interp)

for fname in mpc_files:
    cte_interp = interpolate_cte(os.path.join(folder_path, fname), time_grid)
    if cte_interp is not None:
        mpc_cte_all.append(cte_interp)

pid_cte_array = np.vstack(pid_cte_all)
mpc_cte_array = np.vstack(mpc_cte_all)

pid_cte_mean = np.nanmean(pid_cte_array, axis=0)
mpc_cte_mean = np.nanmean(mpc_cte_array, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(time_grid, pid_cte_mean, label="PID Avg CTE", linewidth=2)
plt.plot(time_grid, mpc_cte_mean, label="MPC Avg CTE", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Cross Track Error [m]")
plt.title("Average CTE vs Time (Clipped to Shortest Run)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def interpolate_heading_error(filepath, time_grid):
    try:
        df = pd.read_csv(filepath)
        time_vals = df["time"].values
        heading_vals = df["heading_error_rad"].values
        interp_fn = interp1d(time_vals, heading_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        return interp_fn(time_grid)
    except:
        return None

pid_heading_all = []
mpc_heading_all = []

for fname in pid_files:
    heading_interp = interpolate_heading_error(os.path.join(folder_path, fname), time_grid)
    if heading_interp is not None:
        pid_heading_all.append(heading_interp)

for fname in mpc_files:
    heading_interp = interpolate_heading_error(os.path.join(folder_path, fname), time_grid)
    if heading_interp is not None:
        mpc_heading_all.append(heading_interp)

pid_heading_array = np.vstack(pid_heading_all)
mpc_heading_array = np.vstack(mpc_heading_all)

pid_heading_mean = np.nanmean(pid_heading_array, axis=0)
mpc_heading_mean = np.nanmean(mpc_heading_array, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(time_grid, pid_heading_mean, label="PID Avg Heading Error", linewidth=2)
plt.plot(time_grid, mpc_heading_mean, label="MPC Avg Heading Error", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Heading Error [rad]")
plt.title("Average Heading Error vs Time (Clipped to Shortest Run)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def interpolate_steer(filepath, time_grid):
    try:
        df = pd.read_csv(filepath)
        time_vals = df["time"].values
        steer_vals = df["steer"].values
        interp_fn = interp1d(time_vals, steer_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        return interp_fn(time_grid)
    except:
        return None

pid_steer_all = []
mpc_steer_all = []

for fname in pid_files:
    steer_interp = interpolate_steer(os.path.join(folder_path, fname), time_grid)
    if steer_interp is not None:
        pid_steer_all.append(steer_interp)

for fname in mpc_files:
    steer_interp = interpolate_steer(os.path.join(folder_path, fname), time_grid)
    if steer_interp is not None:
        mpc_steer_all.append(steer_interp)

pid_steer_array = np.vstack(pid_steer_all)
mpc_steer_array = np.vstack(mpc_steer_all)

pid_steer_mean = np.nanmean(pid_steer_array, axis=0)
mpc_steer_mean = np.nanmean(mpc_steer_array, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(time_grid, pid_steer_mean, label="PID Avg Steering", linewidth=2)
plt.plot(time_grid, mpc_steer_mean, label="MPC Avg Steering", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Steering Command")
plt.title("Average Steering Command vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Interpolation Function for Throttle and Brake
def interpolate_column(filepath, col, time_grid):
    try:
        df = pd.read_csv(filepath)
        time_vals = df["time"].values
        y_vals = df[col].values
        interp_fn = interp1d(time_vals, y_vals, kind='linear', bounds_error=False, fill_value=np.nan)
        return interp_fn(time_grid)
    except:
        return None

pid_throttle_all = []
pid_brake_all = []

for fname in pid_files:
    throttle_interp = interpolate_column(os.path.join(folder_path, fname), "throttle", time_grid)
    brake_interp = interpolate_column(os.path.join(folder_path, fname), "brake", time_grid)
    if throttle_interp is not None: pid_throttle_all.append(throttle_interp)
    if brake_interp is not None: pid_brake_all.append(brake_interp)

mpc_throttle_all = []
mpc_brake_all = []

for fname in mpc_files:
    throttle_interp = interpolate_column(os.path.join(folder_path, fname), "throttle", time_grid)
    brake_interp = interpolate_column(os.path.join(folder_path, fname), "brake", time_grid)
    if throttle_interp is not None: mpc_throttle_all.append(throttle_interp)
    if brake_interp is not None: mpc_brake_all.append(brake_interp)

pid_throttle_mean = np.nanmean(np.vstack(pid_throttle_all), axis=0)
pid_brake_mean = np.nanmean(np.vstack(pid_brake_all), axis=0)
mpc_throttle_mean = np.nanmean(np.vstack(mpc_throttle_all), axis=0)
mpc_brake_mean = np.nanmean(np.vstack(mpc_brake_all), axis=0)

plt.figure(figsize=(10, 6))
plt.plot(time_grid, pid_throttle_mean, label="PID Throttle", linestyle="-", linewidth=2)
plt.plot(time_grid, mpc_throttle_mean, label="MPC Throttle", linestyle="--", linewidth=2)
plt.plot(time_grid, pid_brake_mean, label="PID Brake", linestyle="-", linewidth=2)
plt.plot(time_grid, mpc_brake_mean, label="MPC Brake", linestyle="--", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Command Value")
plt.title("Average Throttle and Brake vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Collect average speed per run for PID
pid_speed_all = []

for fname in pid_files:
    try:
        df = pd.read_csv(os.path.join(folder_path, fname)).dropna()
        time_vals = df["time"].values
        if time_vals[-1] < time_grid[-1]:
            continue

        interp_fn = interp1d(time_vals, df["speed"].values, kind='linear', bounds_error=False, fill_value=np.nan)
        speed_interp = interp_fn(time_grid)  # already in km/h

        pid_speed_all.append(speed_interp)
    except:
        continue

# Collect average speed per run for MPC
mpc_speed_all = []

for fname in mpc_files:
    try:
        df = pd.read_csv(os.path.join(folder_path, fname)).dropna()
        time_vals = df["time"].values
        if time_vals[-1] < time_grid[-1]:
            continue

        interp_fn = interp1d(time_vals, df["speed"].values, kind='linear', bounds_error=False, fill_value=np.nan)
        speed_interp = interp_fn(time_grid)

        mpc_speed_all.append(speed_interp)
    except:
        continue

# Compute average speed per controller
pid_avg_speed = np.nanmean([np.nanmean(run) for run in pid_speed_all])
mpc_avg_speed = np.nanmean([np.nanmean(run) for run in mpc_speed_all])

# Print results
print(f"PID Avg Speed: {pid_avg_speed:.2f} km/h")
print(f"MPC Avg Speed: {mpc_avg_speed:.2f} km/h")

# Lane Exit Detection Setup
lane_boundary = 1.5  # meters
pid_exit_counts = []
mpc_exit_counts = []

# PID lane exits
for fname in pid_files:
    try:
        df = pd.read_csv(os.path.join(folder_path, fname)).dropna()
        exit_mask = df["cte"].abs() > lane_boundary
        num_exits = exit_mask.sum()
        pid_exit_counts.append(num_exits)
    except:
        continue

# MPC lane exits
for fname in mpc_files:
    try:
        df = pd.read_csv(os.path.join(folder_path, fname)).dropna()
        exit_mask = df["cte"].abs() > lane_boundary
        num_exits = exit_mask.sum()
        mpc_exit_counts.append(num_exits)
    except:
        continue

# Compute averages
pid_exit_avg = np.mean(pid_exit_counts)
mpc_exit_avg = np.mean(mpc_exit_counts)

# Print Results
print(f"PID Avg Lane Exits per Run: {pid_exit_avg:.2f}")
print(f"MPC Avg Lane Exits per Run: {mpc_exit_avg:.2f}")


df = pd.read_csv("C:/CARLA/CARLA_0.9.144/pid_data_120.csv")

# Ideal Path Plot
plt.figure(figsize=(10, 6))
plt.plot(df["x_ideal"], df["y_ideal"], color='purple', linewidth=2)
plt.title("Ideal Trajectory (Lane Center)")
plt.xlabel("X Position [m]")
plt.ylabel("Y Position [m]")
plt.grid(True)
plt.axis('equal')  # Keep aspect ratio so radius can be estimated visually
plt.tight_layout()
plt.show()