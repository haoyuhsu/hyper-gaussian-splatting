from plyfile import PlyData, PlyElement

p = "/home/yenchi/Desktop/cs543-final/gaussian-splatting/output/1ec7f-limit-pts=10000-noDense/point_cloud/iteration_30000/point_cloud.ply"
p = "/home/yenchi/Desktop/cs543-final/gaussian-splatting/output/1ec7f-limit-pts=4096-max10000-Dense/point_cloud/iteration_30000/point_cloud.ply"
plydata = PlyData.read(p)
print(plydata['vertex'].count)
