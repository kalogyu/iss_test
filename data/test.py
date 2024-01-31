from geo2 import Geo2Location
import carla
import numpy as np
if __name__ == '__main__':
    # >>>>>获取地图并计算转换矩阵
    client=carla.Client('127.0.0.1',2000)
    # world=client.load_world('Town07')
    world=client.get_world()
    map=world.get_map()

    
    # >>>>>获取车辆当前的gnss并打印信息:
    actors=world.get_actors()
    gnss_actor_list = actors.filter('*gnss*') #找到gnss
    gnss_actor = gnss_actor_list[0]
    print(gnss_actor)
    car_longitude=0.0
    car_latitude=0.0
    car_altitude=0.0
    car_transform=None
    def callback(data):
        global car_longitude,car_latitude,car_altitude
        print(f">>>>>>gnss传感器值:data.longitude 经度:{data.longitude},latitude 维度:{data.latitude},altitude 海拔:{data.altitude}")
        car_longitude=data.longitude
        car_latitude=data.latitude
        car_altitude=data.altitude
        car_transform=data.transform
        print(f"x:{car_transform.location.x},y:{car_transform.location.y},z:{car_transform.location.z}")
    gnss_actor.listen(lambda data: callback(data))
    
    def save_numpy(array,name):
        path_txt=f'./transferMatric_G2L/{name}.txt'
        path_np=f'./transferMatric_G2L/{name}.npy'
        np.savetxt(path_txt, array)
        np.save(path_np, array)

    def load_numpy_txt(name):
        path_txt=f'./transferMatric_G2L/{name}.txt'
        # path_np=f'./transferMatric_G2L/{name}.npy'
        return np.loadtxt(path_txt)
        # np.load(path_np)

    def load_numpy_np(name):
        # path_txt=f'./transferMatric_G2L/{name}.txt'
        path_np=f'./transferMatric_G2L/{name}.npy'
        # np.load(path_txt)
        return np.load(path_np)


    times = 0
    while times < 1:
        print('----------------------------------------------------')
        times +=1
        world.tick()
        # >>>>>>获取车辆的carla.Location并打印信息:
        vehicle_actor_list=actors.filter('*vehicle*')
        vehicle_actor=vehicle_actor_list[0]
        vehicle_location=vehicle_actor.get_location()
        print(f">>>>>>车辆坐标读取值:vehicle location x:{vehicle_location.x},y:{vehicle_location.y},z:{vehicle_location.z}")
        
        # >>>>>carla.GeoLocation -> carla.Location:  
        # 使用Geo2Location类转换
        g2l_obj=Geo2Location(map) #通过加入地图map计算转换矩阵
        trans_matrix=g2l_obj.get_matrix() #转换矩阵
        save_numpy(trans_matrix, 'town07')
        matrix_val=load_numpy_txt('town07')
        g2l_obj.set_matrix(matrix_val)
        
        car_gnss=carla.GeoLocation(longitude=car_longitude,latitude=car_latitude,altitude=car_altitude)
        car_location=carla.Location()
        car_location=g2l_obj.transform(car_gnss)
        print(f'>>>>>>转换后值车辆坐标值:car_location x:{car_location.x},y:{car_location.y},z:{car_location.z}')
        print('----------------------------------------------------')
