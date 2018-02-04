import os
import meshlabxml as mlx


def reduce_point():
	# first reduce to around 2000
	num_faces = 1025
	shape = mlx.FilterScript(file_in='/Users/Angel/Desktop/chair000009.off', file_out='reduced_chair.ply', ml_version='2016.12')
	mlx.remesh.simplify(shape, texture=False, faces=1025)
	shape.run_script()
	topology = mlx.files.measure_topology('reduced_chair.ply')

	#while topology['vert_num']>=1024:
	#	num_faces -= 100
	#	reduced_shape = mlx.FilterScript(file_in='reduced_chair.ply', file_out='reduced_chair.ply')
	#	mlx.remesh.simplify(reduced_shape, texture=False, faces=num_faces)
	#	reduced_shape.run_script()
	#	topology=mlx.files.measure_topology('reduced_chair.ply')

	return


if __name__ == '__main__':
	meshlabserver_path = '/Applications/meshlab.app/Contents/MacOS/'
	os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']
	
	# loop all categories
	src_root = '../data/ModelNet10/'
	dst_root = '../data/reducedModelNet10/'

	category_dirs = [d for d in os.listdir(src_root) if not d.startswith('.') and not d.endswith('.txt')]
	for category_dir in category_dirs:
		if not os.path.exists(dst_root + category_dir + '/train/'):
			os.makedirs(dst_root + category_dir + '/train')
		if not os.path.exists(dst_root + category_dir + '/test/'):
			os.makedirs(dst_root + category_dir + '/test/')

		train_files = [f for f in os.listdir(src_root + category_dir + '/train/') if not f.startswith('.')]
		for train_file in train_files:
			file_in = src_root + category_dir + '/train/' + train_file
			file_out = dst_root + category_dir + '/train/' + train_file
			#reduce_point(file_in, file_out, num_faces)
	
		test_files = [f for f in os.listdir(src_root + category_dir + '/test/') if not f.startswith('.')]
		for test_file in test_files:
			file_in = src_root + category_dir + '/test/' + test_file
			file_out = dst_root + category_dir + '/test/' + test_file
			#reduce_point(file_in, file_out, num_faces)
	
	exit(0)


