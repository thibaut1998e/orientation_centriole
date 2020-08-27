import csv
import numpy as np
import imageio
import os

"""code which saves tiff images from a cloud of point and manages some operations on these cloud of 
points such as rotation"""


simu_data_path = 'simu_centriole_10000pts/simu_centriole_10000pts' #folder with the csv files, containing cloud of points
angles_path = 'simu_centriole_10000pts/centriole_tilt.csv' #assiociated angles


def rotation_matrix_x(tetha):
    tetha = 2 * np.pi * tetha / 360
    return np.array([[np.cos(tetha), 0, np.sin(tetha)],
                     [0,    1,      0],
                     [-np.sin(tetha), 0, np.cos(tetha)]])


def read_csv_cloud(csv_file):
    """read a csv file containing cloud of points and returns the corresponding list of points"""
    with open(csv_file) as f:
        csv_reader = csv.reader(f)
        points = []
        for x in csv_reader:
            point = np.array([float(x[0]), float(x[1]), float(x[2])])
            points.append(point)
    return points


def read_csv_angles(csv_file):
    """read csv file containing angles and returns the list of angles"""
    with open(csv_file) as f:
        csv_reader = csv.reader(f)
        angles = csv_reader.__next__()
        angles = [float(a) for a in angles]
    return angles


class CloudOfPoints:
    def __init__(self, points, tetha_y):
        """class representing a cloud of points, points is a list of points (1 point is an array with 3 elements)
        tetha_y is the initial angle with respect to y axis of the cloud of point"""
        self.points = points
        self.tetha_y = tetha_y


    def maxs(self):
        return [max([p[d] for p in self.points]) for d in range(3)]

    def mins(self):
        return [min([p[d] for p in self.points]) for d in range(3)]

    def save_figure(self, save_location, shape=(20,100,100)):
        """save a figure representing the cloud of point, it normalizes the coordinate so that the range of each dimension
        matches the shape given in argument"""
        binary_rep = np.zeros(shape).astype(np.float32)
        mins = self.mins()
        maxs = self.maxs()
        for p in self.points:
            coord = [int((shape[d]-1)*(p[d] - mins[d])/(maxs[d] - mins[d])) for d in range(3)]
            binary_rep[tuple(coord)] = 1
        imageio.mimwrite(save_location, binary_rep)

    def rotation_x(self, tetha):
        """apply a rotation of angle tetha around y axis"""
        res = []
        for p in self.points:
            p_res = p.dot(rotation_matrix_x(tetha))
            res.append(p_res)
        self.points = res
        self.tetha_y = self.tetha_y + tetha
        print('tetha after', self.tetha_y)

    def rotation_x_to(self, tetha_target):
        """apply a rotation of angle tetha around y axis so that the resulting angle is tetha_target"""
        tetha = tetha_target - self.tetha_y
        self.rotation_y(tetha)


angles = read_csv_angles(angles_path)
for i in range(len(angles)):
    print(f'{i+1} : {angles[i]}')
if not os.path.exists('simu_tiffs'):
    os.makedirs('simu_tiffs')
if not os.path.exists('simu_tiffs_90'):
    os.makedirs('simu_tiffs_90')
for file in os.listdir(simu_data_path):
    idx = int(file.split('.')[0][-4:])
    tetha_y = angles[idx-1]
    csv_path = f'{simu_data_path}/{file}'
    points = read_csv_cloud(csv_path)
    cloud = CloudOfPoints(points, tetha_y)
    cloud.save_figure(f'simu_tiffs/simu{idx}.tiff')
    cloud.rotation_x(90)
    cloud.save_figure(f'simu_tiffs_90/simu{idx}_rotation.tiff')


