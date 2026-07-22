# App Review Insights

**AI 驱动的 App Store 用户评论智能分析平台** — 从 iTunes RSS 抓取评论，通过 DeepSeek 大模型自动发现用户关注的主题、生成证据支撑的洞察报告，并输出可追溯的 PRD 与测试用例。

---

## 功能概览

| 阶段 | 模块 | 功能说明 |
|------|------|----------|
| 数据采集 | `app.py` + iTunes RSS | 通过 Apple 官方 RSS Feed 获取 App Store 评论（非爬虫，合规） |
| 数据清洗 | `data_processor.py` | ID + 内容哈希双重去重，Emoji/垃圾过滤，缺失值处理 |
| AI 分析 | `ai_analyzer.py` | 三阶段 LLM 流水线：主题聚类 → 证据洞察 → PRD + 测试用例 |
| 可视化 | `ui/index.html` | SPA 单页应用，Tab 式结果展示，支持证据溯源交互 |

---

## 技术栈

- **后端**: FastAPI + Uvicorn
- **前端**: 原生 HTML/JS + Tailwind CSS CDN
- **AI**: DeepSeek API（OpenAI 兼容接口）
- **数据处理**: Pandas
- **配置管理**: python-dotenv

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目后进入目录
cd D:\WEB

# 创建并激活虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS / Linux

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件（已提供模板，替换为你的真实 Key）：

```ini
HOST=0.0.0.0
PORT=8000

# DeepSeek API（OpenAI 兼容）
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

> 💡 如果没有 DeepSeek API Key，可前往 [platform.deepseek.com](https://platform.deepseek.com) 注册获取。也可以将 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL` 改为任何 OpenAI 兼容的 API 端点。

### 3. 启动后端服务

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

服务启动后，可访问：
- **API 文档（Swagger）**: http://localhost:8000/docs
- **前端界面**: http://localhost:8000

### 4. 使用前端

1. 在浏览器中打开 http://localhost:8000
2. 在左侧控制面板粘贴 App Store 应用链接（例如 `https://apps.apple.com/us/app/.../id1234567890`）
3. 点击「开始分析」按钮
4. 等待数据处理和 AI 分析完成后，在右侧 Tab 页卡中查看结果：
   - **概览** — 评分分布和整体统计
   - **主题分析** — AI 自动发现的主题聚类
   - **核心洞察** — 每个主题下的问题详情，含严重性评级和证据链
   - **PRD** — 基于用户反馈生成的产品需求文档
   - **测试用例** — 可追溯到原始评论的验证用例

---

## AI 使用说明

### 模型选择

本项目使用 **DeepSeek Chat（V4 Pro）** 模型，通过 OpenAI 兼容接口调用。DeepSeek 在中文理解和结构化 JSON 输出方面表现优秀，适合本项目的分析场景。

### AI 流水线（三阶段）

| 阶段 | AI 角色 | 输入 | 输出 |
|------|---------|------|------|
| **主题聚类** | 产品分析师 | 清洗后的评论列表 | 3-8 个动态主题 + 每条评论的主题标签 |
| **洞察生成** | 资深产品分析师 | 按主题分组的评论 | 每个主题 1-3 个核心问题，含严重性、摘要、证据链、样本量、平均评分 |
| **PRD + 测试** | PM + QA Lead | 完整的洞察报告 | 可追溯的 PRD 需求 + 测试用例（每个结论均关联 `trace_review_ids`） |

### 数据溯源 & 防止幻觉

为降低大模型幻觉风险，本项目在设计上要求：

1. **结构化输出** — 所有 AI 响应必须为纯 JSON，不含任何解释性文字或 Markdown 代码块
2. **证据链强制** — 每个洞察必须附带 `evidence`（`review_id` 列表），PRD 需求和测试用例必须附带 `trace_review_ids`
3. **前端可展开** — 在 UI 中，每条洞察/需求/测试用例旁的 `<details>` 区域可直接查看引用的原始评论内容
4. **跨批次合并** — 主题聚类阶段对评论分批处理（每批 60 条），随后按标签合并跨批次主题，避免单批次偏见

### 错误处理

- **认证失败**（401/403）：立即终止，不进行无意义的重试
- **API 超时/网络错误**：最多重试 3 次，指数退避（2s → 4s → 8s）
- **JSON 解析失败**：支持 Markdown 代码块剥离和括号匹配修复作为降级策略
- **空响应检测**：API 返回空内容时立即报错，不做无效的 JSON 解析

---

## 项目结构

```
D:\WEB\
├── app.py                  # FastAPI 后端入口
├── data_processor.py       # 数据清洗 & 预处理
├── ai_analyzer.py          # AI 分析引擎（三阶段流水线）
├── requirements.txt        # Python 依赖
├── .env                    # 环境变量（API Key 等，不提交到 Git）
├── README.md               # 本文件
├── data/                   # 数据缓存目录（生成文件，不提交到 Git）
│   ├── raw_reviews.json           # RSS 原始数据
│   ├── cleaned_reviews.json       # 清洗后数据
│   ├── reviews_with_topics.json   # 主题标注后数据
│   ├── insights.json              # AI 洞察报告
│   ├── prd.json                   # 产品需求文档
│   └── test_cases.json            # 测试用例
└── ui/
    └── index.html           # SPA 前端
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/collect_reviews` | 采集评论 → 清洗 → AI 分析 → 返回汇总 |
| `POST` | `/api/analyze` | 同上（前端首选别名） |
| `GET` | `/api/data/{filename}` | 读取 `data/` 下的 JSON 文件 |
| `GET` | `/api/summary` | 仪表盘聚合摘要 |
| `GET` | `/` | 前端 SPA |

---

## 离线查看

项目所有分析结果均已缓存至 `data/` 目录。无需网络连接即可：
- 启动服务器 → 打开前端 → 通过 `/api/summary` 和 `/api/data/*` 查看已有的分析结果
- 直接打开 `data/` 目录下的 JSON 文件查看结构化数据
