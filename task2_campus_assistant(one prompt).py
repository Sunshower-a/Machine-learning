import sys
from datetime import datetime

import dashscope
from dashscope import Generation

from config import get_api_key, MODEL_NAME

SYSTEM_PROMPT = (
    '请总结以下通知'
)
TEST_INPUTS = [
    ('课程或作业通知',
     '本周五前提交高数作业，作业包括第三章习题1、3、5，请写在作业本上，下周一课前交给学习委员。'),
]

RESULT_FILE = '测试结果.md'


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
