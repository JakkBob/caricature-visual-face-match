# 跨模态漫画-视觉人脸识别系统

一个用于漫画人脸图像与真实人脸图像匹配的跨模态识别系统，基于深度学习技术实现。

## 项目简介

本项目是硕士研究生毕业论文项目，研究方向为跨模态漫画-视觉人脸识别。系统实现了漫画人脸图像与真实人脸图像的正确匹配，支持双向检索（漫画→真实人脸、真实人脸→漫画）。

## 主要功能

- **跨模态匹配**：支持漫画→真实人脸和真实人脸→漫画的双向匹配
- **批量处理**：支持批量图像上传和匹配
- **实验评估**：提供完整的评估指标（Rank-1/5/10、mAP、Precision、Recall、F1-Score）
- **数据管理**：便捷的数据集管理界面
- **可视化展示**：直观的匹配结果展示和系统状态监控

## 技术栈

- **前端**：Next.js 16 + React 19 + Tailwind CSS + shadcn/ui
- **后端**：Next.js API Routes
- **深度学习**：PyTorch（需用户提供模型文件）

## 项目结构

```
├── config/                    # 配置文件目录
│   └── model.config.ts        # 模型参数配置
├── models/                    # 模型文件目录（需用户放置）
│   ├── face_detector.pt       # 人脸检测模型
│   ├── feature_extractor.pt   # 特征提取模型
│   └── cross_modal_matcher.pt # 跨模态匹配模型
├── data/                      # 数据目录
│   ├── faces/                 # 真实人脸图像
│   └── caricatures/           # 漫画人脸图像
├── src/
│   ├── app/                   # Next.js App Router
│   │   ├── api/               # API路由
│   │   └── page.tsx           # 主页面
│   ├── components/            # React组件
│   │   └── face-match/        # 人脸匹配相关组件
│   ├── lib/                   # 工具库
│   │   ├── models/            # 模型接口
│   │   └── face-match.ts      # 核心匹配逻辑
│   └── types/                 # TypeScript类型定义
└── reports/                   # 评估报告输出目录
```

## 快速开始

### 1. 安装依赖

```bash
bun install
```

### 2. 配置模型

1. 将您的PyTorch模型文件放置到 `models/` 目录：
   - `face_detector.pt` - 人脸检测模型
   - `feature_extractor.pt` - 特征提取模型
   - `cross_modal_matcher.pt` - 跨模态匹配模型（可选）

2. 修改 `config/model.config.ts` 中的模型参数：
   - 模型路径
   - 输入尺寸
   - 特征维度
   - GPU设置等

### 3. 准备数据

将测试图像放置到 `data/` 目录：
- `data/faces/` - 真实人脸图像
- `data/caricatures/` - 漫画人脸图像

### 4. 启动服务

```bash
bun run dev
```

访问 http://localhost:3000 使用系统。

## 模型接口说明

系统预留了以下模型接口，您需要根据实际模型实现具体逻辑：

### 人脸检测器 (`src/lib/models/face-detector.ts`)

```typescript
interface IFaceDetector {
  loadModel(modelPath: string): Promise<boolean>;
  detect(image: Buffer | string): Promise<FaceDetectionResult>;
  detectBatch(images: (Buffer | string)[]): Promise<FaceDetectionResult[]>;
  dispose(): Promise<void>;
}
```

### 特征提取器 (`src/lib/models/feature-extractor.ts`)

```typescript
interface IFeatureExtractor {
  loadModel(modelPath: string): Promise<boolean>;
  extract(image: Buffer | string, modality: ImageModality): Promise<FeatureExtractionResult>;
  extractBatch(images: (Buffer | string)[], modality: ImageModality): Promise<FeatureExtractionResult[]>;
  getFeatureDimension(): number;
  dispose(): Promise<void>;
}
```

### 跨模态匹配器 (`src/lib/models/cross-modal-matcher.ts`)

```typescript
interface ICrossModalMatcher {
  loadModel(modelPath: string): Promise<boolean>;
  computeSimilarity(feature1: number[], feature2: number[]): Promise<number>;
  search(queryFeature: number[], targetFeatures: FeatureVector[], topK?: number): Promise<MatchedPair[]>;
  match(queryFeature: FeatureVector, targetFeatures: FeatureVector[]): Promise<MatchResult>;
  dispose(): Promise<void>;
}
```

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/upload` | POST | 上传单张图像 |
| `/api/upload` | PUT | 批量上传图像 |
| `/api/match` | POST | 执行跨模态匹配 |
| `/api/match` | PUT | 计算两张图像相似度 |
| `/api/evaluate` | POST | 执行评估 |
| `/api/evaluate` | GET | 获取评估历史 |
| `/api/dataset` | GET | 获取数据集信息 |
| `/api/dataset` | DELETE | 删除数据集文件 |
| `/api/config` | GET | 获取配置信息 |
| `/api/status` | GET | 获取系统状态 |

## 配置说明

所有模型参数在 `config/model.config.ts` 中集中管理：

```typescript
export const ModelConfig = {
  faceDetector: {
    modelPath: './models/face_detector.pt',
    inputSize: { width: 640, height: 640 },
    confidenceThreshold: 0.5,
    useGPU: true,
    // ...
  },
  featureExtractor: {
    modelPath: './models/feature_extractor.pt',
    inputSize: { width: 224, height: 224 },
    featureDim: 512,
    // ...
  },
  crossModalMatcher: {
    similarityMethod: 'cosine',
    matchThreshold: 0.5,
    topK: 10,
    // ...
  },
  // ...
};
```

## 评估指标

系统支持以下评估指标：

- **Rank-K 准确率**：Rank-1、Rank-5、Rank-10
- **平均精度**：mAP (Mean Average Precision)
- **分类指标**：Precision、Recall、F1-Score
- **混淆矩阵**：TP、FP、TN、FN

## 开发说明

```bash
# 代码检查
bun run lint

# 数据库操作
bun run db:push
bun run db:generate
```

## 许可证

本项目仅供学术研究使用。

## 作者

硕士研究生 - 跨模态漫画-视觉人脸识别研究方向
