import carla
import random
import time
import numpy as np
import math
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import os

# Constants
TARGET_SPEED_KMH = 160.0
SIMULATION_STEP = 0.05
WAYPOINT_LOOKAHEAD = 12.0 # For yaw error calculation

# PID Controller Gains
STEER_KP = 0.6
STEER_KI = 0.005
STEER_KD = 0.20

THROTTLE_KP = 0.5
THROTTLE_KI = 0.02
THROTTLE_KD = 0.01

# CTE Correction
CTE_CORRECTION_LOOKAHEAD = 8.0
CTE_CORRECTION_WEIGHT = 0.7

CAM_INTERP_SPEED = 0.3
CAM_DISTANCE_BACK = 4
CAM_HEIGHT = 3

# Plotting Constant
PLOT_ENABLED = True
PLOT_TRAJ_FIG_SIZE_INCHES = (5, 7)
MAP_CARLA_X_BOUNDS = (-500, 500)
MAP_CARLA_Y_BOUNDS = (-450, 450)
PLOT_LIVE_CONTROLS_FIG_SIZE_INCHES = (8, 12)
PLOT_POST_ANALYSIS_FIG_SIZE_INCHES = (10, 9)

END_LOOP_PROXIMITY_THRESHOLD = 15.0
MIN_LOOP_TIME_BEFORE_END_CHECK = 60.0

# Global Variables For Plotting
fig_trajectory = None; ax_trajectory = None; actual_traj_line = None; ideal_traj_line = None
current_pos_marker_actual_traj = None; current_pos_marker_ideal_traj = None
actual_path_plot_x_traj = []; actual_path_plot_y_traj = []
ideal_path_plot_x_traj = []; ideal_path_plot_y_traj = []

fig_live_controls = None; axs_live_controls = None
line_speed_actual = None; line_speed_target = None; line_throttle = None
line_steer = None; line_brake = None

sim_times_list = []; throttle_list = []; brake_list = []; steer_list = []
speed_list = []; heading_error_list = []; cte_list = []

target_lane_id = None # Will store the ID of the lane the car should be in

# Helper Functions
def lerp_location(start_loc, end_loc, alpha):
    alpha = max(0.0, min(1.0, alpha))
    return carla.Location(
        x=start_loc.x + (end_loc.x - start_loc.x) * alpha,
        y=start_loc.y + (end_loc.y - start_loc.y) * alpha,
        z=start_loc.z + (end_loc.z - start_loc.z) * alpha,
    )

def get_speed_kmh(vehicle):
    vel = vehicle.get_velocity()
    speed_ms = math.sqrt(vel.x**2 + vel.y**2)
    return speed_ms * 3.6

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
        self.integral_limit = 2.0

    def step(self, error):
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / self.dt
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output

# Plotting Functions
def setup_plots():
    global fig_trajectory, ax_trajectory, actual_traj_line, ideal_traj_line
    global current_pos_marker_actual_traj, current_pos_marker_ideal_traj
    global fig_live_controls, axs_live_controls
    global line_speed_actual, line_speed_target, line_throttle, line_steer, line_brake

    if not PLOT_ENABLED: return
    plt.ion()

    fig_trajectory, ax_trajectory = plt.subplots(figsize=PLOT_TRAJ_FIG_SIZE_INCHES)
    actual_traj_line, = ax_trajectory.plot([], [], 'b-', label='Actual Path', linewidth=1.5)
    ideal_traj_line, = ax_trajectory.plot([], [], 'r--', label='Ideal Lane Center', linewidth=1.0)
    current_pos_marker_actual_traj, = ax_trajectory.plot([], [], 'bo', markersize=5, label='Vehicle')
    current_pos_marker_ideal_traj, = ax_trajectory.plot([], [], 'rx', markersize=5, mew=1.5, label='Ideal WP')
    ax_trajectory.set_xlabel("Y-coordinate (m)")
    ax_trajectory.set_ylabel("X-coordinate (m)")
    ax_trajectory.set_title("PID Live Trajectory")
    ax_trajectory.legend(loc='upper right', fontsize='small')
    ax_trajectory.grid(True)
    ax_trajectory.set_xlim(MAP_CARLA_Y_BOUNDS); ax_trajectory.set_ylim(MAP_CARLA_X_BOUNDS)
    ax_trajectory.set_aspect('equal', adjustable='box')
    fig_trajectory.canvas.manager.set_window_title("PID Trajectory Plot")
    plt.figure(fig_trajectory.number); plt.tight_layout()

    fig_live_controls, axs_live_controls = plt.subplots(4, 1, sharex=True, figsize=PLOT_LIVE_CONTROLS_FIG_SIZE_INCHES)
    fig_live_controls.canvas.manager.set_window_title("PID Live Data")

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
    veh_plot_x = current_vehicle_loc_carla.y; veh_plot_y = current_vehicle_loc_carla.x
    ideal_plot_x = current_ideal_loc_carla.y; ideal_plot_y = current_ideal_loc_carla.x
    actual_path_plot_x_traj.append(veh_plot_x); actual_path_plot_y_traj.append(veh_plot_y)
    ideal_path_plot_x_traj.append(ideal_plot_x); ideal_path_plot_y_traj.append(ideal_plot_y)
    actual_traj_line.set_data(actual_path_plot_x_traj, actual_path_plot_y_traj)
    ideal_traj_line.set_data(ideal_path_plot_x_traj, ideal_path_plot_y_traj)
    current_pos_marker_actual_traj.set_data([veh_plot_x], [veh_plot_y])
    current_pos_marker_ideal_traj.set_data([ideal_plot_x], [ideal_plot_y])
    try: fig_trajectory.canvas.draw_idle()
    except Exception as e: print(f"Trajectory plot error: {e}")

def update_live_data_plots():
    if not PLOT_ENABLED or fig_live_controls is None or not sim_times_list: return
    line_speed_actual.set_data(sim_times_list, speed_list)
    target_speed_y_data = [TARGET_SPEED_KMH] * len(sim_times_list)
    line_speed_target.set_data(sim_times_list, target_speed_y_data)
    line_throttle.set_data(sim_times_list, throttle_list)
    line_steer.set_data(sim_times_list, steer_list)
    line_brake.set_data(sim_times_list, brake_list)
    for ax in axs_live_controls:
        ax.relim(); ax.autoscale_view(scalex=True, scaley=False)
    try: fig_live_controls.canvas.draw_idle()
    except Exception as e: print(f"Live data plot error: {e}")

def generate_post_simulation_plots():
    if not PLOT_ENABLED or not sim_times_list: return
    fig_post_analysis, axs_post = plt.subplots(2, 1, sharex=True, figsize=PLOT_POST_ANALYSIS_FIG_SIZE_INCHES)
    fig_post_analysis.canvas.manager.set_window_title("PID Post-Simulation Analysis")
    axs_post[0].plot(sim_times_list, np.degrees(heading_error_list), 'g-', label='Heading Error (deg)')
    axs_post[0].set_ylabel("Heading Error (deg)"); axs_post[0].legend(loc='upper right', fontsize='small'); axs_post[0].grid(True)
    axs_post[1].plot(sim_times_list, cte_list, 'purple', label='Cross-Track Error (m)')
    axs_post[1].set_ylabel("CTE (m)"); axs_post[1].legend(loc='upper right', fontsize='small'); axs_post[1].grid(True)
    axs_post[1].set_xlabel("Time (s)")
    fig_post_analysis.suptitle("PID Post-Simulation Analysis: Errors")
    plt.figure(fig_post_analysis.number); plt.tight_layout(rect=[0, 0, 1, 0.96])

def save_pid_data_to_csv_pandas(filename=None):
    if filename is None:
        filename = f"pid_data_{int(TARGET_SPEED_KMH)}.csv"
    data = {
        "time": sim_times_list,
        "speed": speed_list,
        "steer": steer_list,
        "throttle": throttle_list,
        "brake": brake_list,
        "cte": cte_list,
        "heading_error_rad": heading_error_list,
        "x_actual": actual_path_plot_y_traj,
        "y_actual": actual_path_plot_x_traj,
        "x_ideal": ideal_path_plot_y_traj,
        "y_ideal": ideal_path_plot_x_traj,
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n PID simulation data saved to: {os.path.abspath(filename)}")

# Main Simulation
client = None
world = None
vehicle = None
original_settings = None
initial_spawn_waypoint_location = None

try:
    print("Connecting to CARLA")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town04')
    world.set_weather(carla.WeatherParameters.ClearNoon)
    print("Connected")

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True; settings.fixed_delta_seconds = SIMULATION_STEP
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    map_ = world.get_map()

    print("Spawning vehicle")
    vehicle_bp = blueprint_library.find('vehicle.mercedes.coupe_2020')
    spawn_points = map_.get_spawn_points()
    spawn_point_transform = spawn_points[5]

    base_wp = map_.get_waypoint(spawn_point_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    initial_spawn_waypoint_location = base_wp.transform.location
    
    current_search_wp = base_wp; leftmost_wp = base_wp
    while True: 
        candidate_further_left = current_search_wp.get_left_lane() 
        if not candidate_further_left or candidate_further_left.lane_type != carla.LaneType.Driving: 
            leftmost_wp = current_search_wp; break 
        current_search_wp = candidate_further_left
    
    second_lane_from_left_wp = leftmost_wp.get_right_lane()
    spawn_wp_final = second_lane_from_left_wp if second_lane_from_left_wp and second_lane_from_left_wp.lane_type == carla.LaneType.Driving else leftmost_wp
    
    target_lane_id = spawn_wp_final.lane_id
    
    if spawn_wp_final == second_lane_from_left_wp: print(f"Targeting second lane from left (ID: {target_lane_id}) for spawning.")
    else: print(f"Could not find valid second lane from left. Spawning in leftmost lane (ID: {target_lane_id}).")
    
    spawn_transform_actual = spawn_wp_final.transform
    spawn_transform_actual.location.z += 0.3

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform_actual)
    if vehicle is None: raise RuntimeError("Failed to spawn vehicle")
    print(f"Vehicle spawned at {vehicle.get_location()} (Target Lane ID: {target_lane_id}), target loop return near X:{initial_spawn_waypoint_location.x:.1f}, Y:{initial_spawn_waypoint_location.y:.1f}")

    if PLOT_ENABLED: setup_plots()

    spectator = world.get_spectator()
    world.tick()

    steering_pid = PIDController(STEER_KP, STEER_KI, STEER_KD, SIMULATION_STEP)
    throttle_pid = PIDController(THROTTLE_KP, THROTTLE_KI, THROTTLE_KD, SIMULATION_STEP)

    print("Starting control loop. Simulation will end if car returns to start after ~" + str(MIN_LOOP_TIME_BEFORE_END_CHECK) + "s.")
    loop_start_time = time.perf_counter(); frame_id_counter = 0; sim_time = 0.0
    prev_carla_location = vehicle.get_transform().location; distance_covered = 0.0

    while True:
        elapsed_loop_internal = time.perf_counter() - loop_start_time
        sleep_for = SIMULATION_STEP - elapsed_loop_internal
        if sleep_for > 0: time.sleep(sleep_for)
        loop_start_time = time.perf_counter()

        world_frame = world.tick(); frame_id_counter += 1; sim_time += SIMULATION_STEP

        vehicle_transform = vehicle.get_transform()
        current_carla_location = vehicle_transform.location
        step_distance = math.sqrt((current_carla_location.x - prev_carla_location.x)**2 +
                                (current_carla_location.y - prev_carla_location.y)**2)
        distance_covered += step_distance; prev_carla_location = current_carla_location

        vehicle_rotation = vehicle_transform.rotation
        forward_vector = vehicle_transform.get_forward_vector()
        current_speed = get_speed_kmh(vehicle)

        cam_target_location = current_carla_location - forward_vector * CAM_DISTANCE_BACK + carla.Location(z=CAM_HEIGHT)
        cam_target_rotation = carla.Rotation(pitch=-15, yaw=vehicle_rotation.yaw, roll=0)
        current_cam_transform = spectator.get_transform()
        new_loc = lerp_location(current_cam_transform.location, cam_target_location, CAM_INTERP_SPEED)
        delta_yaw = cam_target_rotation.yaw - current_cam_transform.rotation.yaw
        if delta_yaw > 180: delta_yaw -= 360
        if delta_yaw < -180: delta_yaw += 360
        new_yaw = current_cam_transform.rotation.yaw + delta_yaw * min(CAM_INTERP_SPEED*2, 1.0)
        new_rot = carla.Rotation(pitch=cam_target_rotation.pitch, yaw=new_yaw, roll=0)
        spectator.set_transform(carla.Transform(new_loc, new_rot))

        # Control Logic
        wp_for_control = None
        if target_lane_id is not None:
            # Get a waypoint on the road near the car, ideally on the target lane
            temp_wp = map_.get_waypoint(current_carla_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if temp_wp:
                if temp_wp.lane_id == target_lane_id:
                    wp_for_control = temp_wp
                else: # Try to find the target lane from temp_wp by hopping
                    search_wp = temp_wp
                    max_hops = 5; hops = 0; found_on_target_lane = False
                    if search_wp.lane_id < target_lane_id: # Need to move right
                        while hops < max_hops and search_wp.lane_id != target_lane_id:
                            r_wp = search_wp.get_right_lane()
                            if r_wp and r_wp.lane_type == carla.LaneType.Driving: search_wp = r_wp
                            else: break
                            hops +=1
                    elif search_wp.lane_id > target_lane_id: # Need to move left
                        while hops < max_hops and search_wp.lane_id != target_lane_id:
                            l_wp = search_wp.get_left_lane()
                            if l_wp and l_wp.lane_type == carla.LaneType.Driving: search_wp = l_wp
                            else: break
                            hops +=1
                    if search_wp.lane_id == target_lane_id:
                        wp_for_control = search_wp
                        found_on_target_lane = True
                    
                    if not found_on_target_lane: # Fallback
                        wp_for_control = temp_wp 
            # If temp_wp is None, wp_for_control remains None, handled below
        else: # Fallback if target_lane_id was somehow not set
             wp_for_control = map_.get_waypoint(current_carla_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        ideal_location_for_plot_carla = current_carla_location
        cte = 0.0; yaw_error_rad = 0.0; cte_correction_angle_rad = 0.0

        if wp_for_control:
            ideal_location_for_plot_carla = wp_for_control.transform.location
            vec_to_veh_3d = current_carla_location - wp_for_control.transform.location
            wp_right_3d = wp_for_control.transform.get_right_vector()
            cte = vec_to_veh_3d.dot(wp_right_3d)
            cte_correction_angle_rad = math.atan2(-cte, CTE_CORRECTION_LOOKAHEAD)
            
            next_wps = wp_for_control.next(WAYPOINT_LOOKAHEAD)
            if not next_wps:
                print("End of road segment"); vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0)); break
            
            target_wp_carla = next_wps[0]
            target_loc_carla = target_wp_carla.transform.location
            vec_to_target = target_loc_carla - current_carla_location
            target_yaw_rad = math.atan2(vec_to_target.y, vec_to_target.x)
            current_yaw_rad = math.radians(vehicle_rotation.yaw)
            yaw_error_rad = math.atan2(math.sin(target_yaw_rad - current_yaw_rad), math.cos(target_yaw_rad - current_yaw_rad))
        else:
            print("Lost road. Stopping vehicle."); vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0)); break

        combined_steer_error = yaw_error_rad + (CTE_CORRECTION_WEIGHT * cte_correction_angle_rad)
        speed_error = TARGET_SPEED_KMH - current_speed
        steer_cmd = steering_pid.step(combined_steer_error)
        throttle_cmd = throttle_pid.step(speed_error)

        steer_cmd = np.clip(steer_cmd, -1.0, 1.0)
        throttle_cmd = np.clip(throttle_cmd, 0.0, 1.0)
        brake_cmd = 0.0

        if speed_error < -10: # More aggressive braking for overspeed
             brake_cmd = np.clip(-throttle_pid.step(speed_error*0.15), 0.0, 0.5) # Stronger braking response
             throttle_cmd = 0.0
        elif abs(steer_cmd) > 0.5 and current_speed > TARGET_SPEED_KMH * 0.4: # Adjusted threshold for hard steer braking
            brake_cmd = np.clip(abs(steer_cmd) * 0.15, 0.0, 0.3) # Gentler braking for steering
            throttle_cmd = min(throttle_cmd, 0.4)
        elif (abs(cte) > 2.0 or abs(math.degrees(yaw_error_rad)) > 30) and current_speed > 70:
            brake_cmd = 0.5
            throttle_cmd = 0.0
            print("Braking: Off-track detected")

        control = carla.VehicleControl(throttle=throttle_cmd, steer=steer_cmd, brake=brake_cmd)
        vehicle.apply_control(control)

        sim_times_list.append(sim_time); throttle_list.append(throttle_cmd); brake_list.append(brake_cmd)
        steer_list.append(steer_cmd); speed_list.append(current_speed)
        heading_error_list.append(yaw_error_rad); cte_list.append(cte)

        if PLOT_ENABLED:
            update_trajectory_plot(current_carla_location, ideal_location_for_plot_carla)
            update_live_data_plots()
            if frame_id_counter % 5 == 0: plt.pause(0.00001)

        print(f"T:{sim_time:6.1f}s | D:{distance_covered:6.0f}m | Spd:{current_speed:5.1f} | Steer:{steer_cmd:+5.2f} | Thr:{throttle_cmd:.2f} | Brk:{brake_cmd:.2f} | CTE:{cte:5.2f} | YawErr:{math.degrees(yaw_error_rad):+5.1f}° | CTE_Ang:{math.degrees(cte_correction_angle_rad):+5.1f}°", end='\r')

        if sim_time > MIN_LOOP_TIME_BEFORE_END_CHECK and initial_spawn_waypoint_location:
            dist_to_start_xy = math.sqrt(
                (current_carla_location.x - initial_spawn_waypoint_location.x)**2 +
                (current_carla_location.y - initial_spawn_waypoint_location.y)**2 )
            if dist_to_start_xy < END_LOOP_PROXIMITY_THRESHOLD:
                print(f"\nVehicle returned to start (distance: {dist_to_start_xy:.2f}m). Ending simulation.")
                break

except KeyboardInterrupt: print("\nInterrupted by user. Exiting.")
except Exception as e: print(f"\nAn error occurred: {e}"); traceback.print_exc()

finally:
    if world and original_settings: print("\nRestoring world settings"); world.apply_settings(original_settings)
    if vehicle: print("Destroying vehicle"); vehicle.destroy()
    if PLOT_ENABLED:
        if sim_times_list: print("Generating post-simulation analysis plots"); generate_post_simulation_plots()
        save_pid_data_to_csv_pandas()
        figures_exist = fig_trajectory or fig_live_controls or plt.get_fignums()
        if figures_exist:
            print("\nSimulation ended. Plot windows are now blocking. Close all plot windows to exit.")
            plt.ioff(); plt.show(block=True)
        else: print("No plots were generated or PLOT_ENABLED was false.")
    print("Finished.")