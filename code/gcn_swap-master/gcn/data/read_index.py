import numpy as np

file_content = '/Users/guanhangwu/Downloads/cora/cora.content'
with open(file_content, 'r') as f:
    f_str = f.read()

f_list = f_str.split('\n')
index_list = [f.split('\t') for f in f_list][:-1]

index_file = np.array([int(f[0]) for f in index_list])
index = np.argsort(index_file)
index_file = index_file[index]

feature = np.array([[int(k) for k in f[1:-1]] for f in index_list])
feature = feature[index, :]

label = [f[-1] for f in index_list]
print(set(label))
exit(1)
label_set = set(label)
label_dict = dict()
for i, k in enumerate(list(label_set)):
    label_dict[k] = i
label_index = np.array([label_dict[key] for key in label])

label_index = label_index[index]

n_values = np.amax(label_index) + 1
label_one_hot = np.eye(n_values)[label_index]

np.save('cora_feature.npy', feature)
# np.save('coro_index.npy', label_index)
np.save('cora_label.npy', label_one_hot)
