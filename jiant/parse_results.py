import os
import csv
from collections import defaultdict, OrderedDict

ROOT_DIR = '/home/hanozbhathena/project'
EXPERIMENTS_DIR = 'jiant_models_save'
DIR_PTH = os.path.join(ROOT_DIR, EXPERIMENTS_DIR)
TASKS = {'mnli', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', 'sst', 'mrpc'}
MAX_SIMUL_TASK_COUNT = 2
TASK_ORDER = ['mnli', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', 'sst', 'mrpc']

task_to_evalmetric = {
    'mnli': 'mnli_accuracy',
    'qqp': 'qqp_acc_f1',
    'qnli': 'qnli_accuracy',
    'rte': 'rte_accuracy',
    'sts-b': 'sts-b_corr',
    'cola': 'cola_mcc',
    'sst': 'sst_accuracy',
    'mrpc': 'mrpc_acc_f1',
}


def get_task(input_str):
    for task in TASKS:
        if task in input_str:
            return task
    return ''


def get_score_for_task(line, task):
    splitted = line.split(',')
    metric = task_to_evalmetric[task]
    for elem in splitted:
        if metric in elem:
            break
    score = float(elem.strip().split(':')[-1])
    return score


def write_to_csv(input_dict, file_path):
    csv_columns = ['task'] + TASK_ORDER
    dict_rows = [{**v, **{'task': k}} for k, v in input_dict.items()]
    csv_file = file_path
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_rows:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def main():
    dict_data = defaultdict(dict)
    for subdir, dirs, files in os.walk(DIR_PTH):
        if 'wnli' in subdir:
            continue
        for file in files:
            if file == 'results.tsv':
                filepath = os.path.join(subdir, file)
                with open(filepath, 'r') as f:
                    lines = f.readlines()[-MAX_SIMUL_TASK_COUNT:]
                    tasks_scores = []
                    for line in lines:
                        task = get_task(line)
                        score = get_score_for_task(line, task)
                        tasks_scores.append((task, score))
                print(tasks_scores)

                if len(tasks_scores) < 2:
                    print('Warning! less than two tasks at {0}'.format(subdir))
                    dict_data[tasks_scores[0][0]][tasks_scores[0][0]] = tasks_scores[0][1]
                else:
                    dict_data[tasks_scores[0][0]][tasks_scores[1][0]] = tasks_scores[0][1]
                    dict_data[tasks_scores[1][0]][tasks_scores[0][0]] = tasks_scores[1][1]

    ord_dict_data = OrderedDict()
    for task in TASK_ORDER:
        if task in dict_data:
            ord_dict_data[task] = dict_data[task]
    output_path = os.path.join(ROOT_DIR, EXPERIMENTS_DIR, 'join_results.csv')
    write_to_csv(ord_dict_data, output_path)


if __name__ == "__main__":
    main()
