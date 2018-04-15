import rospy

from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    MIN_THROTTLE = 0.  # Minimum throttle value
    MAX_THROTTLE = 0.2  # Maximum throttle value

    def __init__(self, vehicle_mass, decel_limit, wheel_radius, wheel_base,
                 steer_ratio, max_lat_accel, max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.decel_limit = decel_limit

        self.throttle_controller = PID(
            kp=0.3,
            ki=0.1,
            kd=0.,

            mn=self.MIN_THROTTLE,
            mx=self.MAX_THROTTLE
        )
        self.yaw_controller = YawController(wheel_base=wheel_base, steer_ratio=steer_ratio, min_speed=0.1,
                                            max_lat_accel=max_lat_accel, max_steer_angle=max_steer_angle)
        self.velocity_lpf = LowPassFilter(
            tau=0.5,  # 1 / (2 * pi * tau) - cutoff frequency
            ts=0.02  # sample time
        )

        self.last_time = rospy.get_time()

    def reset(self):
        self.throttle_controller.reset()

    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity):
        """Compute control variables.

        :param proposed_linear_velocity: target linear velocity
        :param proposed_angular_velocity: target angular velocity
        :param current_linear_velocity: current vehicle linear velocity
        :return: (tuple) control variables (throttle, brake, steering)
        """
        filtered_velocity = self.velocity_lpf.filt(current_linear_velocity)

        steering = self.yaw_controller.get_steering(proposed_linear_velocity, proposed_angular_velocity,
                                                    filtered_velocity)

        velocity_error = proposed_linear_velocity - filtered_velocity

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(velocity_error, sample_time)
        brake = 0

        if proposed_linear_velocity == 0. and filtered_velocity < 0.1:
            throttle = 0.
            brake = 400  # N*m - to hold the car in place if we are stopped at a light. Acceleration ~1 m/s^2

        elif throttle < 0.1 and velocity_error < 0:
            throttle = 0
            desired_deceleration = max(velocity_error, self.decel_limit)
            # Braking torque, N*m
            brake = self.vehicle_mass * self.wheel_radius * abs(desired_deceleration)

        return throttle, brake, steering
