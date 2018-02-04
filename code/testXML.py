import os
import meshlabxml as mlx

meshlabserver_path = '/Applications/meshlab.app/Contents/MacOS/'
os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']

# create an orange cube and apply some transformation
orange_cube = mlx.FilterScript(file_out='orange_cube.ply', ml_version='2016.12')
mlx.create.cube(orange_cube, size=[3.0, 4.0, 5.0], center=True, color='orange')
mlx.transform.rotate(orange_cube, axis='x', angle=45)
mlx.transform.rotate(orange_cube, axis='y', angle=45)
mlx.transform.translate(orange_cube, value=(0, 5.0, 0))
orange_cube.run_script()

