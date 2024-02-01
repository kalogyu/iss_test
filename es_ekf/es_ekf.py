import carla
import numpy as np
import matplotlib.pyplot as plt
from data import Data
from utils import StampedData
from rotations import Quaternion, rpy_jacobian_axis_angle, skew_symmetric
import carla
import numpy as np
from measurement_update import measurement_update

class CarlaDataCollector:
    def __init__(self, world):
        self.world = world
        self.spawn_points = None
        self.ego_vehicle = None
        self.gnss_sensor = None
        self.imu_sensor = None

        self.t = []
        self.gt_p = []
        self.gt_v = []
        self.gt_r = []
        self.gnss_data = []
        self.gnss_t = []
        self.imu_f_data = []
        self.imu_w_data = []
        self.imu_t = []

    def setup_carla(self):
        try:

            map = self.world.get_map()
            self.spawn_points = map.get_spawn_points()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.005
            self.world.apply_settings(settings)

            spectator = self.world.get_spectator()
            spectator.set_transform(carla.Transform(self.spawn_points[0].location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            tesla_model_3 = self.world.get_blueprint_library().filter('model3')[0]
            self.ego_vehicle = self.world.spawn_actor(tesla_model_3, self.spawn_points[0])

            gnss_sensor = self.world.get_blueprint_library().find('sensor.other.gnss')
            gnss_sensor.set_attribute('sensor_tick', '0.01')
            gnss_sensor_transform = carla.Transform(carla.Location(x=1.0, z=2.8))
            self.gnss_sensor = self.world.spawn_actor(gnss_sensor, gnss_sensor_transform, attach_to=self.ego_vehicle)
            self.gnss_sensor.listen(lambda data: self.gnss_callback(data))

            imu_sensor = self.world.get_blueprint_library().find('sensor.other.imu')
            imu_sensor.set_attribute('sensor_tick', '0.001')
            imu_sensor_transform = carla.Transform(carla.Location(x=1.0, z=1.8))
            self.imu_sensor = self.world.spawn_actor(imu_sensor, imu_sensor_transform, attach_to=self.ego_vehicle)
            self.imu_sensor.listen(lambda data: self.imu_callback(data))

            self.ego_vehicle.set_autopilot(True)
            self.world.tick()

            self.clear_data()
            self.collect_data()

        except Exception as e:
            print(e)
        finally:
            self.cleanup()

    def gnss_callback(self, data):
        self.gnss_data.append([data.transform.location.x, data.transform.location.y, data.transform.location.z])
        self.gnss_t.append(data.timestamp)

    def imu_callback(self, data):
        self.imu_f_data.append([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z])
        self.imu_w_data.append([data.gyroscope.x, data.gyroscope.y, data.gyroscope.z])
        self.imu_t.append(data.timestamp)

    def collect_data(self):
        DURATION = 100
        total_time_steps = DURATION / self.world.get_settings().fixed_delta_seconds
        for i in range(int(total_time_steps)):
            self.t.append(i * self.world.get_settings().fixed_delta_seconds)
            loc = self.ego_vehicle.get_location()
            veh = self.ego_vehicle.get_velocity()
            ori = self.ego_vehicle.get_transform()
            self.gt_p.append([loc.x, loc.y, loc.z])
            self.gt_v.append([veh.x, veh.y, veh.z])
            self.gt_r.append([ori.rotation.pitch, ori.rotation.yaw, ori.rotation.roll])
            self.world.tick()

    def cleanup(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
        if self.gnss_sensor is not None:
            self.gnss_sensor.destroy()
        if self.imu_sensor is not None:
            self.imu_sensor.destroy()

    def clear_data(self):
        self.t.clear()
        self.gt_p.clear()
        self.gt_r.clear()
        self.gt_v.clear()
        self.gnss_data.clear()
        self.imu_f_data.clear()
        self.imu_w_data.clear()
        self.imu_t.clear()
        self.gnss_t.clear()

    def create_gt_data(self):
        gt_data = Data(t=np.array(self.t), p=np.array(self.gt_p), v=np.array(self.gt_v), r=np.array(self.gt_r))
        return gt_data

    def create_imu_f_data_stamped(self):
        imu_f_data_stamped = StampedData()
        imu_f_data_stamped.t = np.array(self.imu_t)
        imu_f_data_stamped.data = np.array(self.imu_f_data)
        return imu_f_data_stamped


    def create_imu_w_data_stamped(self):
        imu_w_data_stamped = StampedData()
        imu_w_data_stamped.t = np.array(self.imu_t)
        imu_w_data_stamped.data = np.array(self.imu_w_data)
        return imu_w_data_stamped

    def create_gnss_data_stamped(self):
        gnss_data_stamped = StampedData()
        gnss_data_stamped.t = np.array(self.gnss_t)
        gnss_data_stamped.data = np.array(self.gnss_data)
        return gnss_data_stamped

    def get_collected_data(self):
        return {
            't': np.array(self.t),
            'gt_p': np.array(self.gt_p),
            'gt_v': np.array(self.gt_v),
            'gt_r': np.array(self.gt_r),
            'gnss_data': np.array(self.gnss_data),
            'gnss_t': np.array(self.gnss_t),
            'imu_f_data': np.array(self.imu_f_data),
            'imu_w_data': np.array(self.imu_w_data),
            'imu_t': np.array(self.imu_t),
        }
    
if __name__ == '__main__':

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    collector = CarlaDataCollector(world=world)
    collector.setup_carla()

    imu_f_data_stamped = collector.create_imu_f_data_stamped()
    gt_data = collector.create_gt_data()
    imu_w_data_stamped = collector.create_imu_w_data_stamped()
    gnss_data_stamped = collector.create_gnss_data_stamped()

    gnss_t = list(gnss_data_stamped.t)

    p_est = np.zeros([imu_f_data_stamped.data.shape[0], 3])  # position estimates
    v_est = np.zeros([imu_f_data_stamped.data.shape[0], 3])  # velocity estimates
    q_est = np.zeros([imu_f_data_stamped.data.shape[0], 4])  # orientation estimates as quaternions
    p_cov = np.zeros([imu_f_data_stamped.data.shape[0], 9, 9])  # covariance matrices at each timestep

    p_est[0] = gt_data.p[0]
    v_est[0] = gt_data.v[0]
    q_est[0] = Quaternion(euler=gt_data.r[0]).to_numpy()
    p_cov[0] = np.zeros(9)
    g = np.array([0, 0, -9.81])  # gravity
    l_jac = np.zeros([9, 6])
    l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
    h_jac = np.zeros([3, 9])
    h_jac[:, :3] = np.eye(3)  # measurement model jacobian  

    var_imu_f = 0.10
    var_imu_w = 0.10
    var_gnss  = 0.10
    var_lidar = 2.00

    for k in range(100, imu_f_data_stamped.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
        delta_t = imu_f_data_stamped.t[k] - imu_f_data_stamped.t[k-1]

        # 1. Update state with IMU inputs
        q_prev = Quaternion(*q_est[k - 1, :]) # previous orientation as a quaternion object
        q_curr = Quaternion(axis_angle=(imu_w_data_stamped.data[k - 1]*delta_t)) # current IMU orientation
        c_ns = q_prev.to_mat() # previous orientation as a matrix
        f_ns = (c_ns @ imu_f_data_stamped.data[k - 1]) + g # calculate sum of forces
        p_check = p_est[k - 1, :] + delta_t*v_est[k - 1, :] + 0.5*(delta_t**2)*f_ns
        v_check = v_est[k - 1, :] + delta_t*f_ns
        q_check = q_prev.quat_mult_left(q_curr)

        # 1.1 Linearize the motion model and compute Jacobians
        f_jac = np.eye(9) # motion model jacobian with respect to last state
        f_jac[0:3, 3:6] = np.eye(3)*delta_t
        f_jac[3:6, 6:9] = -skew_symmetric(c_ns @ imu_f_data_stamped.data[k - 1])*delta_t

        # 2. Propagate uncertainty
        q_cov = np.zeros((6, 6)) # IMU noise covariance
        q_cov[0:3, 0:3] = delta_t**2 * np.eye(3)*var_imu_f
        q_cov[3:6, 3:6] = delta_t**2 * np.eye(3)*var_imu_w
        p_cov_check = f_jac @ p_cov[k - 1, :, :] @ f_jac.T + l_jac @ q_cov @ l_jac.T

        # 3. Check availability of GNSS and LIDAR measurements
        if imu_f_data_stamped.t[k] in gnss_data_stamped.t:
            gnss_i = gnss_data_stamped.t.index(imu_f_data_stamped.t[k])
            p_check, v_check, q_check, p_cov_check = \
                measurement_update(var_gnss, p_cov_check, gnss_data_stamped.data[gnss_i], p_check, v_check, q_check)

        # Update states (save)
        p_est[k, :] = p_check
        v_est[k, :] = v_check
        q_est[k, :] = q_check
        p_cov[k, :, :] = p_cov_check

        error_fig, ax = plt.subplots(2, 3)
        error_fig.suptitle('Error Plots')
        num_gt = gt_data.p.shape[0]
        p_est_euler = []
        p_cov_euler_std = []

        from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion
        # Convert estimated quaternions to euler angles
        for i in range(len(q_est)):
            qc = Quaternion(*q_est[i, :])
            p_est_euler.append(qc.to_euler())

            # First-order approximation of RPY covariance
            J = rpy_jacobian_axis_angle(qc.to_axis_angle())
            p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

        p_est_euler = np.array(p_est_euler)
        p_cov_euler_std = np.array(p_cov_euler_std)

        # Get uncertainty estimates from P matrix
        p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

        titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
        for i in range(3):
            ax[0, i].plot(range(num_gt), gt_data.p[:, i] - p_est[:num_gt, i])
            ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
            ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
            ax[0, i].set_title(titles[i])
        ax[0,0].set_ylabel('Meters')

        for i in range(3):
            ax[1, i].plot(range(num_gt), \
                angle_normalize(gt_data.r[:, i] - p_est_euler[:num_gt, i]))
            ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
            ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
            ax[1, i].set_title(titles[i+3])
        ax[1,0].set_ylabel('Radians')
        plt.show()
