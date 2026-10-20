## 项目结构

```text
api_homework/
├── .env                      # 存放 API Key
├── .env.example              # 配置模板
├── .gitignore
├── config.py                 # 统一读取 API Key 的配置模块
├── task1_api_call.py         # 任务一：完成一次 API 调用
├── task2_campus_assistant.py # 任务二：校园信息助手
├── test_inputs.txt           # 任务二的三类测试输入
└── 测试结果.md               # 运行 --batch 后自动生成
```

## 运行环境

- Python 3.12
- 依赖：`dashscope'

## 运行前配置（重要）

1. 到 [阿里云百炼控制台](https://bailian.console.aliyun.com/) 开通模型服务并创建 API-KEY（新用户有免费额度）。
2. 打开项目里的 `.env` 文件，填入你的 Key：

   ```text
   DASHSCOPE_API_KEY=sk-你的真实密钥
   ```

3. **Key 只存放在 `.env` 或系统环境变量中，不写入代码，不提交 GitHub**（`.gitignore` 已排除 `.env`）。

## 所用平台与模型

| 项目 | 内容 |
|---|---|
| 平台 | 阿里云百炼（DashScope） |
| 模型 | `qwen-plus`（通义千问） |
| 调用方式 | dashscope Python SDK |

## 运行方法

```bash
# 任务一：发送一条消息并打印回答
python task1_api_call.py

# 任务二：交互模式（手动输入文本）
python task2_campus_assistant.py

# 任务二：批量模式（自动跑 3 类测试输入，结果保存到 测试结果.md）
python task2_campus_assistant.py --batch
```

## 任务二输出格式

输入任意校园相关文本，模型固定按以下三行格式回答：

```text
一句话总结：
关键信息：
待办事项：
```

已测试的三类输入：

1. 课程或作业通知（"本周五前提交高数作业……"）
2. 活动或讲座通知（"周六下午三点将在图书馆报告厅举办人工智能讲座……"）
3. 学习中的技术问题（"Python 中列表和元组有什么区别？"）

对应的输入与输出见 `测试结果.md`。
