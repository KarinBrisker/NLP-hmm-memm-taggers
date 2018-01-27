import sys

import additional as add
import numpy as np
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


def MEMMClassifier(line, loaded_model, dict, input_size):
    output = ''
    length = len(line)
    line.append('*n*')
    line.append('*n*')
    line.insert(0, '*n*')
    line.insert(0, '*n*')
    possible_keys = [['STRT'], ['STRT']]
    possible_keys_back = [{'STRT':{'STRT':0}}, {'STRT':{'STRT':0}}]
    V_i_t_r = []
    V_i_t_r.append({})
    V_i_t_r[0]['STRT*STRT'] = 0
    BP_i_t_r = []
    BP_i_t_r.append({})
    for i in range(1, length + 1):
        word = line[i + 1]
        is_word_unique = not dict.has_key('w=' + word)
        word_prev = line[i]
        word_prev_prev = line[i - 1]
        word_next = line[i + 2]
        word_next_next = line[i + 3]
        params = [word, word_prev, word_prev_prev, word_next, word_next_next, None, None]
        V_i_t_r.append({})
        BP_i_t_r.append({})
        next_possible_keys = {}
        possible_keys_back.append({})
        for t in possible_keys[i]:
            best_scores = []
            best_tags = None
            params[5] = t
            for t_tag in possible_keys_back[i][t]:
                params[6] = t_tag
                features = add.turnWordToFeatures(params, is_word_unique)
                features_vec = add.convert_features_to_vec(features, dict, input_size)
                probs = loaded_model.predict_proba(features_vec)
                # if t_tag + '-' + t not in V_i_t_r[i-1]:
                #     continue
                scores = V_i_t_r[i-1][t_tag + '*' + t] + np.log(probs)
                scores = scores[0]
                if len(best_scores) == 0:
                    best_scores = scores
                    best_tags = []
                    for j in range(len(best_scores)):
                        best_tags.append(t_tag)
                else:
                    for j in range(len(best_scores)):
                        if best_scores[j] < scores[j]:
                            best_scores[j] = scores[j]
                            best_tags[j] = t_tag
            for j in range(2):
                index = np.argmax(best_scores)
                r = dict[str(index)]
                next_possible_keys[r] = 0
                if r not in possible_keys_back[i + 1]:
                    possible_keys_back[i + 1][r] = {}
                possible_keys_back[i + 1][r][t] = 0
                best_score = best_scores[index]
                V_i_t_r[i][t + '*' + r] = best_score
                BP_i_t_r[i - 1][t + '*' + r] = best_tags[index]
                best_scores[index] = - float('inf')
        possible_keys.append(next_possible_keys.keys())
    best_key = None
    best_score = - float('inf')
    for key in V_i_t_r[length]:
        if V_i_t_r[length][key] > best_score:
            best_key = key
            best_score = V_i_t_r[length][key]
    before_last, last = best_key.split('*')
    if (length > 1):
        output = line[length] + '/' + before_last + ' ' + line[length + 1] + '/' + last
    else:
        output = line[length + 1] + '/' + last
    for i in range(length - 3, -1, -1):
        before_before_last = BP_i_t_r[i + 2][before_last + '*' + last]
        output = line[i + 2] + '/' + before_before_last + ' ' + output
        last = before_last
        before_last = before_before_last
    return output


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
        file_out.write(MEMMClassifier(line, loaded_model, dict, input_size))
        file_out.write('\n')
        counter += 1
        if counter % chunk == 0:
            print float(counter) / size, 'complete'
    file_out.close()
    # TODO: MY GREEDY GOT accuracy = 0.940125133983
    # print 'accuracy =', add.calc_accuracy(add.read_data(out_file_name), add.read_data('dev'))
