import sys

import additional as add
from sklearn.externals import joblib


def read_test_set(file_name):
    data = []
    for line in file(file_name):
        words = line.strip().split(" ")
        data.append(words)
    first_line = data[0]
    all_have_slash = True
    for word in first_line:
        if word.__contains__('/'):
            all_have_slash = False
            break
    if not all_have_slash:
        new_data = []
        for line in data:
            new_line = []
            for word_tag in line:
                new_line.append(word_tag.rsplit('/', 1)[0])
            new_data.append(new_line)
        return new_data
    return data


def GreedyClassifier(line, loaded_model, dict, input_size):
    output = ''
    line.append('*n*')
    line.append('*n*')
    line.insert(0, '*n*')
    line.insert(0, '*n*')
    tag_prev_prev = 'STRT'
    tag_prev = 'STRT'
    for i in range(2, len(line)-2):
        word = line[i]
        is_word_unique = not dict.has_key('w=' + word)
        word_prev = line[i - 1]
        word_prev_prev = line[i - 2]
        word_next = line[i + 1]
        word_next_next = line[i + 2]
        params = [word, word_prev, word_prev_prev, word_next, word_next_next, tag_prev, tag_prev_prev]

        features = add.turnWordToFeatures(params, is_word_unique)
        features_vec = add.convert_features_to_vec(features, dict, input_size)

        tag = loaded_model.predict(features_vec)
        best_tag = dict[str(int(tag))]
        output += line[i] + '/' + best_tag + ' '
        tag_prev_prev = tag_prev
        tag_prev = best_tag
    return output.strip()


def from_map_to_dict(feature_map_file):
    d = {}
    for line in file(feature_map_file):
        (key, val) = line.strip().split(" ")
        d[key] = val
    return d


if __name__ == '__main__':
    input_file_name = sys.argv[1]
    modelname = sys.argv[2]
    feature_map_file = sys.argv[3]
    out_file_name = sys.argv[4]
    dict = from_map_to_dict(feature_map_file)
    loaded_model = joblib.load(modelname)
    input_size =  loaded_model.coef_.shape[1]

    test_set = read_test_set(input_file_name)

    file_out = open(out_file_name, 'w')
    counter = 0
    size = len(test_set)
    chunk = size / 20
    for line in test_set:
        file_out.write(GreedyClassifier(line, loaded_model, dict, input_size))
        file_out.write('\n')
        counter += 1
        if counter % chunk == 0:
            print float(counter) / size, 'complete'
    file_out.close()
    # TODO: MY GREEDY GOT accuracy = 0.940125133983
    # print 'accuracy =', add.calc_accuracy(add.read_data(out_file_name), add.read_data('ass1-tagger-test'))
