import additional as mle
import sys


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


def HMMClassifier(line, score_creator):
    output = ''
    length = len(line)
    possible_keys = [['START'], ['START']]
    for word in line:
        possible_keys.append(emodel.get_tags_for(word))
    V_i_t_r = []
    V_i_t_r.append({})
    V_i_t_r[0]['START*START'] = 0
    BP_i_t_r = []
    BP_i_t_r.append({})
    for i in range(1, length + 1):
        word = line[i - 1]
        V_i_t_r.append({})
        BP_i_t_r.append({})
        for t in possible_keys[i]:
            for r in possible_keys[i + 1]:
                best_score = - float('inf')
                best_tag = None
                for t_tag in possible_keys[i - 1]:
                    score = V_i_t_r[i-1][t_tag + '*' + t] + score_creator.getScore(word, r, t, t_tag)
                    if score > best_score:
                        best_score = score
                        best_tag = t_tag
                V_i_t_r[i][t + '*' + r] = best_score
                BP_i_t_r[i - 1][t + '*' + r] = best_tag
    best_key = None
    best_score = - float('inf')
    for key in V_i_t_r[length]:
        if V_i_t_r[length][key] > best_score:
            best_key = key
            best_score = V_i_t_r[length][key]
    before_last, last = best_key.split('*')
    if (length > 1):
        output = line[length - 2] + '/' + before_last + ' ' + line[length - 1] + '/' + last
    else:
        output = line[length - 1] + '/' + last
    for i in range(length - 3, -1, -1):
        before_before_last = BP_i_t_r[i + 2][before_last + '*' + last]
        output = line[i] + '/' + before_before_last + ' ' + output
        last = before_last
        before_last = before_before_last
    return output


qmodel = mle.QModel()
emodel = mle.EModel()
if __name__ == '__main__':
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    out_file_name = sys.argv[4]

    qmodel.load_from_file(q_mle_filename)
    print 'read q from file'
    emodel.load_from_file(e_mle_filename)
    print 'read e from file'

    scorer = mle.HMMSCORE(qmodel,emodel)

    test_set = read_test_set(input_file_name)

    file_out = open(out_file_name, 'w')
    counter = 0
    size = len(test_set)
    chunk = size / 20
    for line in test_set:
        file_out.write(HMMClassifier(line, scorer))
        file_out.write('\n')
        counter += 1
        if counter % chunk == 0:
            print float(counter) / size, 'complete'
    file_out.close()

    # print 'accuracy =', mle.calc_accuracy(mle.read_data(out_file_name), mle.read_data('ass1-tagger-test'))
