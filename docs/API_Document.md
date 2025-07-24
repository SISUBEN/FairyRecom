
**响应示例：**
```json
{
  "success": true,
  "data": {
    "user_id": 123,
    "algorithm": "hybrid",
    "recommendations": [
      {
        "video_id": 456,
        "score": 0.95,
        "rank": 1
      }
    ],
    "total_count": 10
  },
  "message": "成功为用户 123 生成 10 个推荐"
}
```

### POST /api/recommend/batch

批量为多个用户生成推荐。

**请求体：**
```json
{
  "user_ids": [1, 2, 3],
  "algorithm": "hybrid",
  "n": 10
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "algorithm": "hybrid",
    "total_users": 3,
    "successful_users": 3,
    "results": {
      "1": [
        {
          "video_id": 456,
          "score": 0.95,
          "rank": 1
        }
      ]
    }
  }
}
```

## 🔍 相似度计算

### GET /api/similarity/users/{user_id}

查找与指定用户相似的其他用户。

**路径参数：**
- `user_id` (int): 用户ID

**查询参数：**
- `method` (string): 相似度计算方法，可选值：
  - `cosine`: 余弦相似度（默认）
  - `pearson`: 皮尔逊相关系数
  - `jaccard`: 杰卡德相似度
- `n` (int): 返回数量，默认10

**响应示例：**
```json
{
  "success": true,
  "data": {
    "user_id": 123,
    "method": "cosine",
    "similar_users": [
      {
        "user_id": 456,
        "similarity": 0.85,
        "rank": 1
      }
    ],
    "total_count": 50
  }
}
```

### GET /api/similarity/videos/{video_id}

查找与指定视频相似的其他视频。

**路径参数：**
- `video_id` (int): 视频ID

**查询参数：**
- `method` (string): 相似度计算方法，可选值：
  - `cosine`: 余弦相似度（默认）
  - `pearson`: 皮尔逊相关系数
- `n` (int): 返回数量，默认10

**响应示例：**
```json
{
  "success": true,
  "data": {
    "video_id": 123,
    "method": "cosine",
    "similar_videos": [
      {
        "video_id": 456,
        "similarity": 0.92,
        "rank": 1
      }
    ],
    "total_count": 30
  }
}
```

## 💡 推荐解释

### GET /api/explain/{user_id}/{video_id}

解释为什么向特定用户推荐某个视频。

**路径参数：**
- `user_id` (int): 用户ID
- `video_id` (int): 视频ID

**响应示例：**
```json
{
  "success": true,
  "data": {
    "explanation": "基于您对相似视频的高评分..."
  },
  "message": "推荐解释生成成功"
}
```

## 📊 统计信息

### GET /api/stats/user/{user_id}

获取用户的详细统计信息。

**响应示例：**
```json
{
  "success": true,
  "data": {
    "user_id": 123,
    "profile": {
      "total_videos": 50,
      "avg_affinity_score": 0.75,
      "avg_rating": 4.2,
      "like_rate": 0.8,
      "avg_sentiment": 0.65
    },
    "watched_videos": [
      {
        "video_id": 456,
        "affinity_score": 0.95
      }
    ],
    "total_watched": 50
  }
}
```

### GET /api/stats/video/{video_id}

获取视频的详细统计信息。

**响应示例：**
```json
{
  "success": true,
  "data": {
    "video_id": 123,
    "features": {
      "total_users": 100,
      "avg_affinity_score": 0.7,
      "avg_rating": 4.0,
      "like_rate": 0.75,
      "avg_sentiment": 0.6
    },
    "popularity_rank": 15,
    "viewers": [
      {
        "user_id": 456,
        "affinity_score": 0.9
      }
    ],
    "total_viewers": 100
  }
}
```

### GET /api/stats/system

获取整个系统的统计信息。

**响应示例：**
```json
{
  "success": true,
  "data": {
    "total_users": 1000,
    "total_videos": 500,
    "total_interactions": 50000,
    "avg_user_interactions": 50.0,
    "avg_video_interactions": 100.0,
    "sparsity": 0.9,
    "top_videos": [
      {
        "video_id": 123,
        "popularity_score": 0.95
      }
    ],
    "most_active_users": [
      {
        "user_id": 456,
        "interaction_count": 200
      }
    ]
  }
}
```

## 😊 情感分析

### POST /api/sentiment/analyze

分析文本的情感倾向。

**请求体（单个文本）：**
```json
{
  "text": "这个视频真的很棒！"
}
```

**请求体（批量文本）：**
```json
{
  "texts": ["文本1", "文本2", "文本3"]
}
```

**响应示例（单个文本）：**
```json
{
  "success": true,
  "data": {
    "text": "这个视频真的很棒！",
    "sentiment_score": 0.8,
    "sentiment_label": "positive",
    "confidence": 0.6
  },
  "message": "情感分析完成"
}
```

**响应示例（批量文本）：**
```json
{
  "success": true,
  "data": {
    "total_texts": 3,
    "analyzed_texts": 3,
    "results": [
      {
        "index": 0,
        "text": "文本1",
        "sentiment_score": 0.7,
        "sentiment_label": "positive",
        "confidence": 0.4
      }
    ],
    "summary": {
      "positive_count": 2,
      "negative_count": 0,
      "neutral_count": 1,
      "avg_sentiment_score": 0.65
    }
  }
}
```

### POST /api/sentiment/batch_reviews

批量分析用户评论的情感。

**请求体：**
```json
{
  "user_id": 123,
  "video_id": 456,
  "reviews": [
    {
      "review_id": 1,
      "content": "评论内容1"
    },
    {
      "review_id": 2,
      "content": "评论内容2"
    }
  ]
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "user_id": 123,
    "video_id": 456,
    "total_reviews": 2,
    "analyzed_reviews": 2,
    "results": [
      {
        "review_id": 1,
        "content": "评论内容1",
        "sentiment_score": 0.8,
        "sentiment_label": "positive",
        "confidence": 0.6
      }
    ],
    "summary": {
      "positive_count": 1,
      "negative_count": 0,
      "neutral_count": 1,
      "avg_sentiment_score": 0.65,
      "overall_sentiment": "positive"
    }
  }
}
```

## 🗄️ 数据库管理

### POST /api/database/use_ext_db

切换到外部数据库。

**请求体：**
```json
{
  "db_path": "/path/to/external/database.db"
}
```

## ❌ 错误代码

| HTTP状态码 | 错误类型 | 说明 |
|-----------|---------|------|
| 400 | Bad Request | 请求参数错误 |
| 404 | Not Found | API接口不存在 |
| 405 | Method Not Allowed | HTTP方法不允许 |
| 500 | Internal Server Error | 服务器内部错误 |

## 🚀 快速开始

1. **启动服务器：**
   ```bash
   python -m src.fairyrecom.api.recommendation_api
   ```

2. **健康检查：**
   ```bash
   curl http://localhost:5000/api/health
   ```

3. **获取推荐：**
   ```bash
   curl "http://localhost:5000/api/recommend/user/1?algorithm=hybrid&n=5"
   ```

4. **情感分析：**
   ```bash
   curl -X POST http://localhost:5000/api/sentiment/analyze \
        -H "Content-Type: application/json" \
        -d '{"text": "这个视频很棒！"}'
   ```

## 📝 注意事项

1. **数据限制：**
   - 推荐数量：1-100
   - 批量推荐用户数：≤100
   - 批量情感分析文本数：≤100

2. **情感分析评分规则：**
   - 0.0-0.4: 负面情感
   - 0.4-0.6: 中性情感
   - 0.6-1.0: 正面情感

3. **推荐算法说明：**
   - `collaborative_filtering`: 基于用户行为相似性
   - `content_based`: 基于内容特征相似性
   - `popularity`: 基于热门程度
   - `hybrid`: 结合多种算法
   - `diversified`: 增加推荐多样性
   - `cold_start`: 针对新用户的推荐
