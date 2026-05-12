# System ID 索引越界 Bug 修复

## 修复日期
2026-04-17

## 问题描述
训练时出现 CUDA 索引越界错误：
```
C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\cuda\Indexing.cu:1308:
block: [42,0,0], thread: [32,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
...
RuntimeError: CUDA error: device-side assert triggered
```

错误发生在模型的 `system_embed` Embedding 层。

## 根本原因

### 数据格式问题
数据文件中的 `system_id` 使用 **GNSS 系统的 ASCII 码**：
- 67 = 'C' (BDS/北斗)
- 71 = 'G' (GPS)
- 82 = 'R' (GLONASS)
- 69 = 'E' (Galileo)
- 等等...

### 模型限制
模型中的 Embedding 层定义为：
```python
self.system_embed = nn.Embedding(10, system_emb_dim, padding_idx=0)
```
只支持索引范围 **0-9**，但数据中的 system_id 值为 **67-82**，导致索引越界。

## 修复方案

### 添加 system_id 映射函数
在 `data/dataset.py` 中添加映射函数，将 ASCII 码映射到 0-9 范围：

```python
def map_system_id_to_index(system_ids: np.ndarray) -> np.ndarray:
    """
    将 GNSS system_id (ASCII 码) 映射到模型可用的索引 (0-9)。

    映射规则：
        0: padding (保留)
        1: GPS (G=71)
        2: GLONASS (R=82)
        3: Galileo (E=69)
        4: BDS (C=67)
        5: QZSS (J=74)
        6: IRNSS (I=73)
        7-9: 保留给未来系统或未知系统
    """
    mapping = {
        71: 1,  # G -> GPS
        82: 2,  # R -> GLONASS
        69: 3,  # E -> Galileo
        67: 4,  # C -> BDS
        74: 5,  # J -> QZSS
        73: 6,  # I -> IRNSS
    }

    mapped_ids = np.zeros_like(system_ids)
    for ascii_code, idx in mapping.items():
        mapped_ids[system_ids == ascii_code] = idx

    # 处理未知系统
    unknown_mask = (mapped_ids == 0) & (system_ids != 0)
    if unknown_mask.any():
        unknown_ids = np.unique(system_ids[unknown_mask])
        print(f"[Warning] Unknown system_id values: {unknown_ids}")
        mapped_ids[unknown_mask] = 7

    return mapped_ids
```

### 在数据加载时应用映射
修改 `STECEpochDataset.__getitem__()` 方法：

```python
# 提取字段
system_ids = df["system_id"].values.astype(np.int64)
# 映射 system_id 从 ASCII 码到模型索引 (0-9)
system_ids = map_system_id_to_index(system_ids)
```

## 验证结果

### 映射测试
```python
test_ids = np.array([67, 71, 67, 71, 0])
mapped = map_system_id_to_index(test_ids)

# 结果：
# Original: [67, 71, 67, 71, 0]
# Mapped:   [4,  1,  4,  1,  0]
```

### 索引范围检查
- 原始范围：67-82 (超出 Embedding 范围)
- 映射后范围：0-9 (符合 Embedding 要求)
- Padding 值：0 → 0 (保持不变)

## 影响范围

### 修改的文件
- `data/dataset.py`: 添加映射函数并在数据加载时应用

### 不需要修改的文件
- `data/collate.py`: padding 处的 system_ids 已经是 0，无需修改
- `models/transformer.py`: Embedding 层定义保持不变
- `configs/default.yaml`: 配置文件无需修改

## GNSS 系统编码对照表

| ASCII 码 | 字符 | GNSS 系统 | 映射索引 |
|---------|------|----------|---------|
| 0       | -    | Padding  | 0       |
| 71      | G    | GPS      | 1       |
| 82      | R    | GLONASS  | 2       |
| 69      | E    | Galileo  | 3       |
| 67      | C    | BDS      | 4       |
| 74      | J    | QZSS     | 5       |
| 73      | I    | IRNSS    | 6       |
| 其他    | -    | Unknown  | 7       |

## 注意事项

### 数据兼容性
- 如果数据文件中出现新的 GNSS 系统，会自动映射到索引 7 并打印警告
- 建议在训练前检查数据中的所有 system_id 值

### 模型兼容性
- 旧模型的 checkpoint 仍然可用，因为 Embedding 层的定义没有改变
- 只是输入数据的索引值发生了变化

### 性能影响
- 映射操作是向量化的，性能开销可忽略不计
- 不影响训练速度

## 测试建议

运行训练前，建议先检查数据中的 system_id 范围：
```python
import pandas as pd
from pathlib import Path

files = list(Path('model_stations').glob('*.csv'))[:10]
all_system_ids = set()
for f in files:
    df = pd.read_csv(f)
    all_system_ids.update(df['system_id'].unique())

print(f'Unique system_id values: {sorted(all_system_ids)}')
print(f'ASCII interpretation: {[chr(x) for x in sorted(all_system_ids)]}')
```

## 下一步

修复完成后，可以正常运行训练：
```bash
python scripts/train.py
```

如果仍然遇到问题，请检查：
1. 数据文件中是否有其他异常的 system_id 值
2. role_type 的值是否在 0-2 范围内
3. CUDA 环境是否正常
