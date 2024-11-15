import pandas as pd
import json
import argparse

file_path = 'few_shot/train-00000-of-00001-fe8894d41b7815be.parquet'


def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    answers = []

    for row in df.iterrows():
        answer = {}
        answer['question'] = row[1]['question']
        answer['sql'] = row[1]['SQL']
        answers.append(answer)

    return answers


def main(cu):
    answers = read_parquet(file_path)
    if cu == 1:
        with open('few_shot/dev.json', 'r') as f:
            exa = json.load(f)
        for temp in exa:
            answer = {}
            answer['question'] = temp['question']
            answer['sql'] = temp['SQL']
            answers.append(answer)

    with open('few_shot/QA.json', 'w') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()

    # 添加命令行选项

    ## 这里的dataset是ppl_dev.json
    parser.add_argument("--cu", type=int, default=1)
    # 解析命令行参数
    args = parser.parse_args()

    main(args.cu)
