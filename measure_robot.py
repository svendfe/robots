"""
Script to measure the actual wheel parameters from the CoppeliaSim model.
Run this while the simulation is loaded (but can be paused).
"""
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def main():
    print('*** Connecting to CoppeliaSim...')
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    robot_id = 'PioneerP3DX'
    
    # Get handles
    robot_base = sim.getObject(f'/{robot_id}')
    left_motor = sim.getObject(f'/{robot_id}/leftMotor')
    right_motor = sim.getObject(f'/{robot_id}/rightMotor')
    
    # Try to get wheel handles
    try:
        left_wheel = sim.getObject(f'/{robot_id}/leftWheel')
        right_wheel = sim.getObject(f'/{robot_id}/rightWheel')
        has_wheels = True
    except:
        print("Could not find wheel objects, will use motor positions")
        has_wheels = False
    
    print("\n" + "="*60)
    print("ROBOT MEASUREMENTS")
    print("="*60)
    
    # Get robot base position
    base_pos = sim.getObjectPosition(robot_base, sim.handle_world)
    print(f"\nRobot base position: ({base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f})")
    
    # Get motor positions (relative to world)
    left_motor_pos = sim.getObjectPosition(left_motor, sim.handle_world)
    right_motor_pos = sim.getObjectPosition(right_motor, sim.handle_world)
    
    print(f"\nLeft motor position:  ({left_motor_pos[0]:.4f}, {left_motor_pos[1]:.4f}, {left_motor_pos[2]:.4f})")
    print(f"Right motor position: ({right_motor_pos[0]:.4f}, {right_motor_pos[1]:.4f}, {right_motor_pos[2]:.4f})")
    
    # Calculate wheel base (distance between motors)
    wheel_base = np.sqrt(
        (right_motor_pos[0] - left_motor_pos[0])**2 +
        (right_motor_pos[1] - left_motor_pos[1])**2
    )
    print(f"\n>>> WHEEL_BASE (distance between motors): {wheel_base:.4f} m")
    
    if has_wheels:
        # Get wheel bounding box to estimate radius
        try:
            left_wheel_pos = sim.getObjectPosition(left_wheel, sim.handle_world)
            print(f"\nLeft wheel position: ({left_wheel_pos[0]:.4f}, {left_wheel_pos[1]:.4f}, {left_wheel_pos[2]:.4f})")
            
            # Wheel radius can be estimated from wheel center height (assuming wheel touches ground)
            wheel_radius_estimate = left_wheel_pos[2]
            print(f"\n>>> WHEEL_RADIUS (estimated from height): {wheel_radius_estimate:.4f} m")
        except Exception as e:
            print(f"Could not get wheel info: {e}")
    
    # Try to get shape info for more accurate measurements
    print("\n" + "-"*60)
    print("Attempting to get shape bounding boxes...")
    
    # List all objects under the robot
    print("\nObjects in robot hierarchy:")
    try:
        # Get all objects
        objects = sim.getObjectsInTree(robot_base, sim.handle_all, 0)
        for obj in objects:
            name = sim.getObjectAlias(obj, 0)
            pos = sim.getObjectPosition(obj, robot_base)
            print(f"  {name}: relative pos ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    except Exception as e:
        print(f"Could not enumerate objects: {e}")
    
    print("\n" + "="*60)
    print("SUGGESTED VALUES FOR robotica.py:")
    print("="*60)
    print(f"WHEEL_BASE = {wheel_base:.4f}")
    if has_wheels:
        print(f"WHEEL_RADIUS = {wheel_radius_estimate:.4f}  # (estimated)")
    print("="*60)

if __name__ == '__main__':
    main()
