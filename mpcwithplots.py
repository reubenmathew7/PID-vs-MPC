import carla
import numpy as np
import time
import math
import random
from scipy.optimize import minimize
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import os

# Constants
TARGET_SPEED_KMH = 160.0
SIMULATION_STEP = 0.05
MPC_HORIZON = 7
MPC_DT = SIMULATION_STEP
VEHICLE_L = 2.8  # wheelbase
MAX_STEER_RAD = math.radians(45)
MAX_ACCEL = 2.0
MIN_ACCEL = -2.0

# Weights
W_CTE = 5.0
W_HE = 5.0
W_VS = 4
W_STEER = 10.0
W_ACCEL = 1.0
W_DSTEER = 100.0
W_DACCEL = 10.0

# Camera Smoothing
CAM_INTERP_SPEED = 0.3
CAM_DISTANCE_BACK = 3
CAM_HEIGHT = 3.0

# Plotting Constants
PLOT_ENABLED = True
PLOT_TRAJ_FIG_SIZE_INCHES = (5, 7)
MAP_CARLA_X_BOUNDS = (-500, 500)
MAP_CARLA_Y_BOUNDS = (-450, 450)
PLOT_LIVE_CONTROLS_FIG_SIZE_INCHES = (8, 12)
PLOT_POST_ANALYSIS_FIG_SIZE_INCHES = (10, 9)

END_LOOP_PROXIMITY_THRESHOLD = 15.0
MIN_LOOP_TIME_BEFORE_END_CHECK = 60.0 # Seconds
PLOT_UPDATE_INTERVAL = 5
STEER_PLOT_SCALE_LIVE = 1.0

# Global Variables for Plotting
fig_trajectory = None; ax_trajectory = None; actual_traj_line = None; ideal_traj_line = None
current_pos_marker_actual_traj = None; current_pos_marker_ideal_traj = None
actual_path_plot_x_traj = []; actual_path_plot_y_traj = []
ideal_path_plot_x_traj = []; ideal_path_plot_y_traj = []
start_marker = None; end_marker = None

fig_live_controls = None; axs_live_controls = None
line_speed_actual = None; line_speed_target = None; line_throttle = None
line_steer = None; line_brake = None

sim_times_list = []; throttle_list = []; brake_list = []; steer_list = []
speed_list = []; heading_error_list = []; cte_list = []
target_speed_history_plot = []
initial_spawn_waypoint_location = None
simulation_start_time_plots = 0

# Helper Functions
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def get_speed_kmh(vehicle):
    vel = vehicle.get_velocity()
    speed_ms = math.sqrt(vel.x**2 + vel.y**2)
    return speed_ms * 3.6

def lerp_location(start, end, alpha):
    return carla.Location(
        x=start.x + (end.x - start.x) * alpha,
        y=start.y + (end.y - start.y) * alpha,
        z=start.z + (end.z - start.z) * alpha
    )

# Plotting Functions
def setup_plots():
    global fig_trajectory, ax_trajectory, actual_traj_line, ideal_traj_line
    global current_pos_marker_actual_traj, current_pos_marker_ideal_traj
    global fig_live_controls, axs_live_controls
    global line_speed_actual, line_speed_target, line_throttle, line_steer, line_brake
    global start_marker, end_marker

    if not PLOT_ENABLED: return
    plt.ion()

    fig_trajectory, ax_trajectory = plt.subplots(figsize=PLOT_TRAJ_FIG_SIZE_INCHES)
    actual_traj_line, = ax_trajectory.plot([], [], 'b-', label='Actual Path', linewidth=1.5)
    ideal_traj_line, = ax_trajectory.plot([], [], 'r--', label='Ideal Lane Center', linewidth=1.0)
    current_pos_marker_actual_traj, = ax_trajectory.plot([], [], 'bo', markersize=5, label='Vehicle')
    current_pos_marker_ideal_traj, = ax_trajectory.plot([], [], 'rx', markersize=5, mew=1.5, label='Ideal WP')

    ax_trajectory.set_xlabel("Y-coordinate (m)")
    ax_trajectory.set_ylabel("X-coordinate (m)")
    ax_trajectory.set_title("MPC Live Trajectory")
    ax_trajectory.legend(loc='upper right', fontsize='small')
    ax_trajectory.grid(True)
    ax_trajectory.set_xlim(MAP_CARLA_Y_BOUNDS); ax_trajectory.set_ylim(MAP_CARLA_X_BOUNDS)
    ax_trajectory.set_aspect('equal', adjustable='box')
    fig_trajectory.canvas.manager.set_window_title("MPC Trajectory Plot")
    plt.figure(fig_trajectory.number); plt.tight_layout()

    fig_live_controls, axs_live_controls = plt.subplots(4, 1, sharex=True, figsize=PLOT_LIVE_CONTROLS_FIG_SIZE_INCHES)
    fig_live_controls.canvas.manager.set_window_title("MPC Live Data")

    axs_live_controls[0].set_ylabel("Speed (km/h)"); axs_live_controls[0].set_ylim(-5, TARGET_SPEED_KMH * 1.2 + 5); axs_live_controls[0].grid(True)
    line_speed_actual, = axs_live_controls[0].plot([], [], 'b-', label='Actual Speed')
    line_speed_target, = axs_live_controls[0].plot([], [], 'r--', label='Target Speed')
    axs_live_controls[0].legend(loc='upper right', fontsize='small')

    axs_live_controls[1].set_ylabel("Throttle (0-1)"); axs_live_controls[1].set_ylim(-0.05, 1.05); axs_live_controls[1].grid(True)
    line_throttle, = axs_live_controls[1].plot([], [], 'g-', label='Throttle')
    axs_live_controls[1].legend(loc='upper right', fontsize='small')

    axs_live_controls[2].set_ylabel("Steering (-1 to 1)"); axs_live_controls[2].set_ylim(-1.05, 1.05); axs_live_controls[2].grid(True)
    line_steer, = axs_live_controls[2].plot([], [], 'purple', label='Steering')
    axs_live_controls[2].legend(loc='upper right', fontsize='small')

    axs_live_controls[3].set_ylabel("Brake (0-1)"); axs_live_controls[3].set_ylim(-0.05, 1.05); axs_live_controls[3].grid(True)
    line_brake, = axs_live_controls[3].plot([], [], 'orangered', label='Brake')
    axs_live_controls[3].legend(loc='upper right', fontsize='small')

    axs_live_controls[3].set_xlabel("Time (s)")
    fig_live_controls.suptitle("Live Vehicle Data")
    plt.figure(fig_live_controls.number); plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)

def update_trajectory_plot(current_vehicle_loc_carla, current_ideal_loc_carla):
    if not PLOT_ENABLED or fig_trajectory is None: return

    # Draw lines using the globally populated lists
    if actual_traj_line:
        actual_traj_line.set_data(actual_path_plot_x_traj, actual_path_plot_y_traj)

    # ideal_path_plot_x_traj and ideal_path_plot_y_traj are now fully populated
    if ideal_traj_line:
        ideal_traj_line.set_data(ideal_path_plot_x_traj, ideal_path_plot_y_traj)

    # Update current position markers using the passed arguments
    # Vehicle's current position marker
    if current_pos_marker_actual_traj:
        veh_plot_x = current_vehicle_loc_carla.y
        veh_plot_y = current_vehicle_loc_carla.x
        current_pos_marker_actual_traj.set_data([veh_plot_x], [veh_plot_y])

    # Ideal waypoint marker
    if current_pos_marker_ideal_traj:
        if current_ideal_loc_carla:
            ideal_plot_x = current_ideal_loc_carla.y
            ideal_plot_y = current_ideal_loc_carla.x
            current_pos_marker_ideal_traj.set_data([ideal_plot_x], [ideal_plot_y])
        else:
            current_pos_marker_ideal_traj.set_data([np.nan], [np.nan])

    try:
        fig_trajectory.canvas.draw_idle()
    except Exception as e:
        print(f"Trajectory plot error: {e}")

def update_live_data_plots():
    if not PLOT_ENABLED or fig_live_controls is None or not sim_times_list: return
    line_speed_actual.set_data(sim_times_list, speed_list)
    line_speed_target.set_data(sim_times_list, target_speed_history_plot)
    line_throttle.set_data(sim_times_list, throttle_list)
    line_steer.set_data(sim_times_list, steer_list)
    line_brake.set_data(sim_times_list, brake_list)
    for ax_ts_single in axs_live_controls:
        ax_ts_single.relim(); ax_ts_single.autoscale_view(scalex=True, scaley=False)
    try: fig_live_controls.canvas.draw_idle()
    except Exception as e: print(f"Live data plot error: {e}")

def generate_post_simulation_plots():
    if not PLOT_ENABLED or not sim_times_list: return
    fig_post_analysis, axs_post = plt.subplots(2, 1, sharex=True, figsize=PLOT_POST_ANALYSIS_FIG_SIZE_INCHES)
    fig_post_analysis.canvas.manager.set_window_title("MPC Post-Simulation Analysis")
    axs_post[0].plot(sim_times_list, np.degrees(heading_error_list), 'g-', label='Heading Err (deg)')
    axs_post[0].set_ylabel("Heading Error (deg)"); axs_post[0].legend(loc='upper right', fontsize='small'); axs_post[0].grid(True)
    axs_post[1].plot(sim_times_list, cte_list, 'purple', label='Cross-Track Error (m)')
    axs_post[1].set_ylabel("CTE (m)"); axs_post[1].legend(loc='upper right', fontsize='small'); axs_post[1].grid(True)
    axs_post[1].set_xlabel("Time (s)")
    fig_post_analysis.suptitle("MPC Post-Simulation Analysis: Errors")
    plt.figure(fig_post_analysis.number); plt.tight_layout(rect=[0, 0, 1, 0.96])

def save_mpc_data_to_csv_pandas(filename=None):
    if filename is None:
         filename = f"mpc_data_{int(TARGET_SPEED_KMH)}.csv"
    """Saves simulation data from the MPC controller to a CSV file."""
    if not sim_times_list:
        print("No data to save. Simulation might have ended prematurely or data lists are empty.")
        return

    # Ensure all lists have the same length, take the minimum length if not
    min_len = len(sim_times_list)
    lists_to_check = [
        speed_list, steer_list, throttle_list, brake_list, cte_list,
        heading_error_list, actual_path_plot_y_traj, actual_path_plot_x_traj,
        ideal_path_plot_y_traj, ideal_path_plot_x_traj
    ]
    for lst in lists_to_check:
        min_len = min(min_len, len(lst))

    if min_len == 0:
        print("Warning: One or more data lists are empty after length check. No CSV will be saved.")
        return

    data = {
        "time": sim_times_list[:min_len],
        "speed": speed_list[:min_len],
        "steer": steer_list[:min_len],    
        "throttle": throttle_list[:min_len],
        "brake": brake_list[:min_len],
        "cte": cte_list[:min_len],
        "heading_error_rad": heading_error_list[:min_len],
        "x_actual": actual_path_plot_y_traj[:min_len],
        "y_actual": actual_path_plot_x_traj[:min_len],
        "x_ideal": ideal_path_plot_y_traj[:min_len],
        "y_ideal": ideal_path_plot_x_traj[:min_len],
    }

    if len(ideal_path_plot_y_traj) < min_len:
        data["x_ideal"] = ideal_path_plot_y_traj + [np.nan] * (min_len - len(ideal_path_plot_y_traj))
    if len(ideal_path_plot_x_traj) < min_len:
        data["y_ideal"] = ideal_path_plot_x_traj + [np.nan] * (min_len - len(ideal_path_plot_x_traj))


    df = pd.DataFrame(data)
    try:
        df.to_csv(filename, index=False)
        print(f"\nMPC simulation data saved to: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"\nError saving MPC data to CSV: {e}")


# MPC Controller
class MPCController:
    def __init__(self, N, dt, L):
        self.N = N
        self.dt = dt
        self.L = L
        self.prev_steer = 0.0 
        self.prev_accel = 0.0
    def vehicle_model(self, state, accel, steer):
        x, y, yaw, v = state
        v_for_yaw = max(v, 0.01) 
        x += v * math.cos(yaw) * self.dt
        y += v * math.sin(yaw) * self.dt
        yaw += (v_for_yaw / self.L) * math.tan(steer) * self.dt 
        v += accel * self.dt
        return [x, y, normalize_angle(yaw), max(0, v)]
    def cost_fn(self, u, state, ref_traj): 
        cost = 0.0
        steer_rad_seq = u[1::2] 
        accel_seq = u[0::2]
        st_rad, ac = self.prev_steer, self.prev_accel
        current_pred_state = np.copy(state)
        for i in range(self.N):
            current_pred_state = self.vehicle_model(current_pred_state, accel_seq[i], steer_rad_seq[i])
            rx, ry, ryaw, rv = ref_traj[i] 
            px, py, pyaw, pv = current_pred_state
            cte_val_cost = np.hypot(px - rx, py - ry)
            he_val_cost = normalize_angle(pyaw - ryaw)
            ve = pv - rv
            cost += W_CTE * cte_val_cost**2 + W_HE * he_val_cost**2 + W_VS * ve**2
            cost += W_STEER * steer_rad_seq[i]**2 + W_ACCEL * accel_seq[i]**2
            if i == 0:
                cost += W_DSTEER * (normalize_angle(steer_rad_seq[i] - st_rad))**2 
                cost += W_DACCEL * (accel_seq[i] - ac)**2
            else:
                cost += W_DSTEER * (normalize_angle(steer_rad_seq[i] - steer_rad_seq[i-1]))**2
                cost += W_DACCEL * (accel_seq[i] - accel_seq[i-1])**2
        return cost
    def solve(self, state, ref_traj): 
        u0 = np.zeros(self.N * 2)
        u0[0::2] = self.prev_accel 
        u0[1::2] = self.prev_steer 
        bounds = []
        for _ in range(self.N):
            bounds.append((MIN_ACCEL, MAX_ACCEL))
            bounds.append((-MAX_STEER_RAD, MAX_STEER_RAD)) 
        result = minimize(self.cost_fn, u0, args=(state, ref_traj),
                          bounds=bounds, method='SLSQP',
                          options={'maxiter': 80, 'ftol': 1e-3, 'disp': False})
        optimal_accel = self.prev_accel 
        optimal_steer_rad = self.prev_steer 
        if result.success:
            optimal_accel = np.clip(result.x[0], MIN_ACCEL, MAX_ACCEL)
            optimal_steer_rad = np.clip(result.x[1], -MAX_STEER_RAD, MAX_STEER_RAD)
        else:
            print(f"MPC optimization failed: {result.message}. Using previous controls.")
        self.prev_accel = optimal_accel
        self.prev_steer = optimal_steer_rad 
        carla_steer_cmd = optimal_steer_rad / MAX_STEER_RAD
        return optimal_accel, np.clip(carla_steer_cmd, -1.0, 1.0)

# Reference Generator
def generate_reference(vehicle_loc, current_yaw_rad, target_speed_mps, carla_map_obj, N, dt):
    ref_traj_np = np.zeros((N, 4)) 
    wp = carla_map_obj.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if not wp: 
        print("WARN: generate_reference off-road. Projecting straight.")
        current_x, current_y = vehicle_loc.x, vehicle_loc.y
        for i in range(N):
             dist_step = target_speed_mps * dt * (i + 1)
             ref_traj_np[i, 0] = current_x + dist_step * math.cos(current_yaw_rad)
             ref_traj_np[i, 1] = current_y + dist_step * math.sin(current_yaw_rad)
             ref_traj_np[i, 2] = current_yaw_rad
             ref_traj_np[i, 3] = target_speed_mps
        return ref_traj_np
    temp_wp = wp
    while True: 
        left = temp_wp.get_left_lane()
        if not left or left.lane_type != carla.LaneType.Driving: break
        temp_wp = left
    right = temp_wp.get_right_lane()
    if right and right.lane_type == carla.LaneType.Driving: wp = right
    else: wp = temp_wp 
    current_wp_for_horizon = wp
    for i in range(N):
        lookahead_distance_for_step = target_speed_mps * dt * (i + 1) 
        next_wps_list = current_wp_for_horizon.next(lookahead_distance_for_step) 
        if not next_wps_list:
            if i > 0: 
                prev_ref_x, prev_ref_y, prev_ref_yaw, prev_ref_v = ref_traj_np[i-1,:]
                ref_traj_np[i, 0] = prev_ref_x + prev_ref_v * dt * math.cos(prev_ref_yaw)
                ref_traj_np[i, 1] = prev_ref_y + prev_ref_v * dt * math.sin(prev_ref_yaw)
                ref_traj_np[i, 2] = prev_ref_yaw
                ref_traj_np[i, 3] = prev_ref_v
            else: 
                dist_step_fallback = target_speed_mps * dt
                ref_traj_np[i, 0] = vehicle_loc.x + dist_step_fallback * math.cos(current_yaw_rad)
                ref_traj_np[i, 1] = vehicle_loc.y + dist_step_fallback * math.sin(current_yaw_rad)
                ref_traj_np[i, 2] = current_yaw_rad
                ref_traj_np[i, 3] = target_speed_mps
            continue 
        target_wp_for_step = next_wps_list[0] 
        p = target_wp_for_step.transform
        ref_traj_np[i, 0] = p.location.x
        ref_traj_np[i, 1] = p.location.y
        ref_traj_np[i, 2] = normalize_angle(math.radians(p.rotation.yaw))
        ref_traj_np[i, 3] = target_speed_mps 
    return ref_traj_np

# Main
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    world.set_weather(carla.WeatherParameters.ClearNoon)

    original_settings = world.get_settings() 
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIMULATION_STEP
    world.apply_settings(settings)

    carla_map_obj = world.get_map() 
    blueprints = world.get_blueprint_library()
    vehicle_bp = blueprints.find('vehicle.mercedes.coupe_2020')
    if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute('color', '0,0,255')

    spawn_points = carla_map_obj.get_spawn_points() 
    if not spawn_points or len(spawn_points) <= 5: 
        raise RuntimeError("Map has insufficient spawn points for index 5")
    base_spawn_transform_from_list = spawn_points[5]
    wp_at_spawn = carla_map_obj.get_waypoint(base_spawn_transform_from_list.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if not wp_at_spawn:
        raise RuntimeError(f"Spawn point index 5 ({base_spawn_transform_from_list.location}) is not on a drivable lane")
    initial_spawn_waypoint_location = wp_at_spawn.transform.location 
    temp_wp_lane_select = wp_at_spawn
    while True:
        left_lane = temp_wp_lane_select.get_left_lane()
        if not left_lane or left_lane.lane_type != carla.LaneType.Driving: break
        temp_wp_lane_select = left_lane
    right_lane = temp_wp_lane_select.get_right_lane()
    if right_lane and right_lane.lane_type == carla.LaneType.Driving: final_spawn_wp = right_lane
    else: final_spawn_wp = temp_wp_lane_select 
    spawn_transform_actual = final_spawn_wp.transform 
    spawn_transform_actual.location.z += 0.3 
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform_actual)
    if not vehicle: raise RuntimeError(f"Failed to spawn vehicle at chosen lane: {spawn_transform_actual}")
    print(f"Vehicle spawned at {vehicle.get_location()}")

    if PLOT_ENABLED: setup_plots()

    spectator = world.get_spectator()
    mpc_controller = MPCController(MPC_HORIZON, MPC_DT, VEHICLE_L)

    world.tick()
    neutral_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.1)
    for _ in range(10): 
        vehicle.apply_control(neutral_control)
        world.tick()
    print("Vehicle settled.")

    transform_init = vehicle.get_transform() 
    spectator.set_transform(carla.Transform(transform_init.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
    world.tick()

    distance_covered = 0.0
    prev_location = vehicle.get_transform().location
    frame_id_counter = 0 
    sim_time_plot = 0.0
    simulation_start_time_real = time.time()
    plot_counter = 0

    if PLOT_ENABLED and start_marker: 
        start_marker.set_data([prev_location.y], [prev_location.x])

    while True:
        world.tick() 
        frame_id_counter += 1
        sim_time_plot += SIMULATION_STEP

        current_transform = vehicle.get_transform()
        current_carla_location = current_transform.location
        step_distance = math.sqrt((current_carla_location.x - prev_location.x)**2 + \
                                  (current_carla_location.y - prev_location.y)**2)
        distance_covered += step_distance
        prev_location = current_carla_location

        current_yaw_deg = current_transform.rotation.yaw
        current_yaw_rad = math.radians(current_yaw_deg)
        current_speed_kmh_val = get_speed_kmh(vehicle)
        current_speed_mps_val = current_speed_kmh_val / 3.6
        current_speed_mps_val = max(current_speed_mps_val, 0.01) 

        forward_vec = current_transform.get_forward_vector()
        cam_target_location = current_carla_location - forward_vec * CAM_DISTANCE_BACK + carla.Location(z=CAM_HEIGHT)
        cam_target_rotation = carla.Rotation(pitch=-15, yaw=current_yaw_deg, roll=0)
        current_cam = spectator.get_transform()
        new_loc_cam = lerp_location(current_cam.location, cam_target_location, CAM_INTERP_SPEED)
        delta_yaw_cam = cam_target_rotation.yaw - current_cam.rotation.yaw
        delta_yaw_cam = (delta_yaw_cam + 180) % 360 - 180 
        new_yaw_cam = current_cam.rotation.yaw + delta_yaw_cam * min(CAM_INTERP_SPEED * 2.0, 1.0)
        new_rot_cam = carla.Rotation(pitch=cam_target_rotation.pitch, yaw=new_yaw_cam, roll=0)
        spectator.set_transform(carla.Transform(new_loc_cam, new_rot_cam))

        current_state_np = np.array([current_carla_location.x, current_carla_location.y, \
                                     current_yaw_rad, current_speed_mps_val])
        target_speed_mps = TARGET_SPEED_KMH / 3.6
        ref_traj_np = generate_reference(current_carla_location, current_yaw_rad, \
                                         target_speed_mps, carla_map_obj, MPC_HORIZON, MPC_DT)

        # Calculate CTE and Heading Error for plotting AND for override condition
        current_ideal_location_for_plot = None
        cte_val_plot = 0.0
        heading_error_val_plot = 0.0
        if ref_traj_np.shape[0] > 0:
            z_val_ideal = current_carla_location.z 
            current_ideal_location_for_plot = carla.Location(x=ref_traj_np[0, 0], y=ref_traj_np[0, 1], z=z_val_ideal)
            
            ref_x_current = current_ideal_location_for_plot.x
            ref_y_current = current_ideal_location_for_plot.y
            ref_yaw_current = ref_traj_np[0, 2] 

            cte_val_plot = -(-(ref_x_current - current_carla_location.x) * math.sin(current_yaw_rad) + \
                           (ref_y_current - current_carla_location.y) * math.cos(current_yaw_rad))
            heading_error_val_plot = normalize_angle(ref_yaw_current - current_yaw_rad)

        # Get MPC Commands
        optimal_accel_cmd, steer_cmd = mpc_controller.solve(current_state_np, ref_traj_np) 

        # Convert MPC Acceleration to Throttle/Brake
        if optimal_accel_cmd >= 0:
            throttle_cmd = np.clip(optimal_accel_cmd / MAX_ACCEL, 0.0, 1.0) if MAX_ACCEL > 0 else 0.0
            brake_cmd = 0.0
        else:
            throttle_cmd = 0.0
            brake_cmd = np.clip(optimal_accel_cmd / MIN_ACCEL, 0.0, 1.0) if MIN_ACCEL < 0 else 0.0
        
        # Safety Brake Override
        override_active_this_tick = False
        if (abs(cte_val_plot) > 2.0 or abs(math.degrees(heading_error_val_plot)) > 30) and current_speed_kmh_val > 70:
            print_throttle_override = throttle_cmd
            throttle_cmd = 0.0
            brake_cmd = 0.5
            override_active_this_tick = True

        # Apply Control to Vehicle
        control = carla.VehicleControl(throttle=float(throttle_cmd), steer=float(steer_cmd), brake=float(brake_cmd))
        vehicle.apply_control(control)

        # Store Data for Plotting
        sim_times_list.append(sim_time_plot)
        speed_list.append(current_speed_kmh_val) 
        target_speed_history_plot.append(TARGET_SPEED_KMH) 
        throttle_list.append(float(throttle_cmd))
        brake_list.append(float(brake_cmd))
        steer_list.append(float(steer_cmd))
        cte_list.append(cte_val_plot)
        heading_error_list.append(heading_error_val_plot)

        # Actual path data
        actual_path_plot_x_traj.append(current_carla_location.y) # CARLA Y for plot X
        actual_path_plot_y_traj.append(current_carla_location.x) # CARLA X for plot Y

        # Ideal path data
        if current_ideal_location_for_plot:
            ideal_path_plot_x_traj.append(current_ideal_location_for_plot.y) # CARLA Y for plot X
            ideal_path_plot_y_traj.append(current_ideal_location_for_plot.x) # CARLA X for plot Y
        else:
            ideal_path_plot_x_traj.append(np.nan)
            ideal_path_plot_y_traj.append(np.nan)

        # Update Plots Periodically
        plot_counter += 1
        if PLOT_ENABLED and plot_counter >= PLOT_UPDATE_INTERVAL:
            plot_counter = 0
            update_trajectory_plot(current_carla_location, current_ideal_location_for_plot)
            update_live_data_plots()
            if frame_id_counter % (PLOT_UPDATE_INTERVAL * 2) == 0: 
                 try: plt.pause(0.00001) 
                 except Exception: pass 

        # Print Telemetry
        override_msg = "OVERRIDE" if override_active_this_tick else ""
        print(f"T:{sim_time_plot:6.1f}s|D:{distance_covered:6.0f}m|Spd:{current_speed_kmh_val:5.1f}|Str:{steer_cmd:+5.2f}|Thr:{throttle_cmd:.2f}|Brk:{brake_cmd:.2f}{override_msg}|MPC_A:{optimal_accel_cmd:+5.2f}|CTE:{cte_val_plot:5.2f}|HE:{math.degrees(heading_error_val_plot):+5.1f}°", end='\r')

        # End Loop Condition
        if sim_time_plot > MIN_LOOP_TIME_BEFORE_END_CHECK and initial_spawn_waypoint_location:
            dist_to_start_xy = math.sqrt(
                (current_carla_location.x - initial_spawn_waypoint_location.x)**2 +
                (current_carla_location.y - initial_spawn_waypoint_location.y)**2 )
            if dist_to_start_xy < END_LOOP_PROXIMITY_THRESHOLD:
                print(f"\nVehicle returned to start (distance: {dist_to_start_xy:.2f}m). Ending simulation.")
                break
        
        # Real-time Pacing
        elapsed_real_time = time.time() - simulation_start_time_real
        expected_sim_time = frame_id_counter * SIMULATION_STEP
        delay = expected_sim_time - elapsed_real_time
        if delay > 0:
            time.sleep(delay)

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    traceback.print_exc()
finally:
    if PLOT_ENABLED:
        if 'vehicle' in locals() and vehicle and vehicle.is_alive and actual_path_plot_x_traj:
            final_loc = vehicle.get_location()
            if end_marker: 
                end_marker.set_data([final_loc.y], [final_loc.x])
            if ax_trajectory: 
                ax_trajectory.relim(); ax_trajectory.autoscale_view()
            if fig_trajectory: 
                try: fig_trajectory.canvas.draw_idle()
                except: pass 
        if sim_times_list:
            print("\nGenerating post-simulation analysis plots")
            generate_post_simulation_plots()
            save_mpc_data_to_csv_pandas()
        figures_exist = plt.get_fignums() 
        if figures_exist:
            print("\nSimulation ended. Plot windows are now blocking. Close all plot windows to exit.")
            plt.ioff(); plt.show(block=True)
        else:
            print("No plots were generated or PLOT_ENABLED was false.")

    if 'world' in locals() and world and 'original_settings' in locals() and original_settings is not None :
        print("\nRestoring world settings")
        current_settings_final = world.get_settings()
        if current_settings_final.synchronous_mode: 
            current_settings_final.synchronous_mode = False
            current_settings_final.fixed_delta_seconds = None 
            world.apply_settings(current_settings_final)
            print("Synchronous mode disabled, fixed_delta_seconds reset.")
        else:
            print("Synchronous mode was already disabled or settings not applied.")

    if 'vehicle' in locals() and vehicle:
        if vehicle.is_alive:
            print("Destroying vehicle")
            vehicle.destroy()
    print("Simulation finished.")
