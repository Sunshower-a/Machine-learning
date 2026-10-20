import sys
from datetime import datetime

import dashscope
from dashscope import Generation

from config import get_api_key, MODEL_NAME

SYSTEM_PROMPT = (
    '你是校园信息助手。用户会给你一段校园相关的文本，'
    '你必须严格按照下面三行的固定格式输出，不要输出任何多余内容：\n'
    '一句话总结：<用一句话概括这段文本>\n'
    '关键信息：<分号分隔的关键要点，如时间、地点、人物、事项>\n'
    '待办事项：<需要去做的事，多条用分号分隔；没有则写"无">'
)

# 三类必测输入：从 test_inputs.txt 读取
def load_test_inputs(path='test_inputs.txt'):
    with open(path, encoding='utf-8') as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    tags = ['课程或作业通知', '活动或讲座通知', '技术问题咨询']
    tags += [f'其他输入{i - len(tags) + 1}' for i in range(len(tags), len(texts))]
    return list(zip(tags, texts))


TEST_INPUTS = load_test_inputs()

RESULT_FILE = '测试结果(more prompt).md'


def ask(text):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': text},
    ]
    response = Generation.call(model=MODEL_NAME, messages=messages,
                               result_format='message')
    if response.status_code == 200:
        return response.output.choices[0].message.content
    return f'[调用失败] {response.code} - {response.message}'


def run_interactive():
    print('=== 校园信息助手（输入 q 退出）===')
    while True:
        text = input('\n请输入校园相关文本 > ').strip()
        if text.lower() == 'q':
            break
        if not text:
            continue
        print('\n' + ask(text))


def run_batch():
    lines = [
        '# 校园信息助手 —— 测试结果',
        '',
        f'- 测试时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'- 所用平台：阿里云百炼（DashScope）　所用模型：{MODEL_NAME}',
        '',
    ]
    print('=== 批量测试开始 ===')
    for i, (tag, text) in enumerate(TEST_INPUTS, 1):
        print(f'\n[{i}/3] {tag}')
        print('输入：' + text)
        output = ask(text)
        print('输出：')
        print(output)
        lines += [
            f'## 测试 {i}：{tag}',
            '',
            '**输入**',
            '',
            '```text',
            text,
            '```',
            '',
            '**输出**',
            '',
            '```text',
            output,
            '```',
            '',
        ]
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n=== 完成，输入与输出已保存到 {RESULT_FILE} ===')


if __name__ == '__main__':
    dashscope.api_key = get_api_key()
    if '--batch' in sys.argv:
        run_batch()
    else:
        run_interactive()
