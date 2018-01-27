import sys
import additional as add


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


def GreedyClassifier(line, score_creator):
    output = ''
    prev = 'START'
    prev_prev = 'START'
    for word in line:
        tags = emodel.get_tags_for(word)
        best_tag = None
        best_score = - float('inf')
        for tag in tags:
            score = score_creator.getScore(word,tag,prev,prev_prev)
            if score > best_score:
                best_score = score
                best_tag = tag
        output += word + '/' + best_tag + ' '
        prev_prev = prev
        prev = best_tag
    return output.strip()


qmodel = add.QModel()
emodel = add.EModel()
if __name__ == '__main__':
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]
    out_file_name = sys.argv[4]

    qmodel.load_from_file(q_mle_filename)
    print 'read q from file'
    emodel.load_from_file(e_mle_filename)
    print 'read e from file'

    scorer = add.HMMSCORE(qmodel,emodel)

    test_set = read_test_set(input_file_name)

    file_out = open(out_file_name, 'w')
    counter = 0
    size = len(test_set)
    chunk = size / 20
    for line in test_set:
        file_out.write(GreedyClassifier(line, scorer))
        file_out.write('\n')
        counter += 1
        if counter % chunk == 0:
            print float(counter) / size, 'complete'
    file_out.close()

    # print 'accuracy =', add.calc_accuracy(add.read_data(out_file_name), add.read_data('ass1-tagger-test'))
