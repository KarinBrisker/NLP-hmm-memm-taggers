import scipy.sparse as sprs
import sys

def read_data(file_name):
    data = []
    for line in file(file_name):
        features = line.strip().split(" ")
        data.append([features[0], features[1:]])
    return data


def convert_features_to_vec(features, feature_dict, input_size):
    llm = sprs.lil_matrix((1, input_size), dtype=float)
    for feature in features:
        if feature in feature_dict:
            llm[0,int(feature_dict[feature])] = 1
    return llm


if __name__ == '__main__':
    # read the program arguments
    features_file = sys.argv[1]
    feature_vecs_file = sys.argv[2]
    feature_map_file = sys.argv[3]

    data_set = read_data(features_file)

    feature_index = 0
    feature_dict = {}
    label_index = 0
    label_dict = {}

    feature_vecs = open(feature_vecs_file, 'w')
    feature_map = open(feature_map_file, 'w')
    for line in data_set:
        label = line[0]
        if label not in label_dict:
            label_dict[label] = label_index
            feature_map.write(str(label_index) + ' ' + label + '\n')
            label_index += 1
        y = label_dict[label]
        sparse = []
        for feature in line[1]:
            if feature not in feature_dict:
                feature_map.write(feature + ' ' + str(feature_index) + '\n')
                feature_dict[feature] = feature_index
                feature_index += 1
            sparse.append(feature_dict[feature])
        sparse.sort()
        feature_vecs.write(str(y))
        for fea in sparse:
            feature_vecs.write(' ' + str(fea) + ':1')
        feature_vecs.write('\n')
    feature_vecs.close()
    feature_map.close()
