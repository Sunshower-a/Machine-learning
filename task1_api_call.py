import dashscope
from dashscope import Generation

from config import get_api_key, MODEL_NAME


def main():
    #从环境变量 / .env 读取 API Key
    dashscope.api_key = get_api_key()

    #构造一条消息
    messages = [
        {'role': 'system', 'content': '你是一个乐于助人的 AI 助手。'},
        {'role': 'user', 'content': '用一句话介绍一下什么是机器学习。'},
    ]

    #调用模型（result_format='message' 指定返回对话格式，
    print(f'正在调用模型 {MODEL_NAME} ...')
    response = Generation.call(model=MODEL_NAME, messages=messages,
                               result_format='message')

    #处理返回结果
    if response.status_code == 200:
        reply = response.output.choices[0].message.content
        print('模型回答：')
        print(reply)
    else:
        print(f'调用失败：{response.code} - {response.message}')


if __name__ == '__main__':
    main()
