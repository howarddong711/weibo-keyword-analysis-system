# 微博关键词分析系统 (Weibo Keyword Analysis System)

这是一个用于分析微博数据的系统，专注于关键词相关的帖子采集、预处理、情感分析和可视化。主要针对“全运会”主题的数据进行分析。

## 项目结构

```
net_space/
├── assets/                 # 资源文件（如字体）
│   └── fonts/
├── data/                   # 数据文件夹
│   ├── html/               # HTML 数据
│   ├── processed/          # 处理后的数据
│   └── raw/                # 原始数据
├── notebooks/              # Jupyter 笔记本
│   └── weibo_data_collection.ipynb  # 数据采集笔记本
├── reports/                # 报告和可视化结果
│   ├── figures/            # 图表文件
│   └── visualization_summary.json  # 可视化总结
├── scripts/                # Python 脚本
│   ├── preprocess_weibo.py # 数据预处理
│   ├── sentiment_analysis.py # 情感分析
│   └── visualize_sentiment.py # 可视化
├── config.json             # 配置文件（关键词、Cookie 等）
├── README.md               # 项目说明
├── requirements.txt        # 依赖列表
└── .gitignore              # Git 忽略文件
```

## 功能模块

### 1. 数据采集 (Data Collection)
- 使用 `notebooks/quanyunhui_data_collection.ipynb` 采集微博数据
- 支持多关键词搜索（如“全运会”、“全运会志愿者”等）
- 采集帖子正文、互动指标和评论

### 2. 数据预处理 (Data Preprocessing)
- 运行 `scripts/preprocess_weibo.py` 清洗和重塑数据
- 处理时间格式、文本清理等

### 3. 情感分析 (Sentiment Analysis)
- 使用 `scripts/sentiment_analysis.py` 进行情感分析
- 支持多种引擎：SnowNLP、Transformers（可选）

### 4. 可视化 (Visualization)
- 使用 `scripts/visualize_sentiment.py` 生成图表和词云
- 输出到 `reports/figures/` 文件夹

## 环境要求

- Python 3.7+
- 依赖包见 `requirements.txt`

## 安装和使用

1. 克隆或下载项目到本地

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 配置参数：
   - 编辑 `config.json` 文件，设置关键词、Cookie 等参数

4. 数据采集：
   - 打开 `notebooks/weibo_data_collection.ipynb`
   - 运行笔记本采集数据

5. 数据预处理：
   ```bash
   python scripts/preprocess_weibo.py
   ```

6. 情感分析：
   ```bash
   python scripts/sentiment_analysis.py
   ```

7. 可视化：
   ```bash
   python scripts/visualize_sentiment.py
   ```

## 配置

编辑 `config.json` 文件来自定义参数：

- `TOPICS`: 关键词列表
- `COOKIE_STR`: 微博 Cookie 字符串
- `HTTP_INTERVAL`: 请求间隔
- `PAGE_DEPTH`: 页面深度
- `MAX_COMMENTS`: 最大评论数
- `MIN_TOTAL_RECORDS` / `MAX_TOTAL_RECORDS`: 记录数范围

## 注意事项

- 数据采集需要有效的微博 Cookie，请在 `config.json` 中填写
- 部分依赖（如 transformers、torch）为可选，用于高级情感分析
- 可视化需要中文字体支持，已包含 Noto Serif CJK 字体

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交 Issue 和 Pull Request