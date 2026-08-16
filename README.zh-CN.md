# StarryVector

[![CI](https://github.com/Hai-Wenxiang/StarryVector/actions/workflows/ci.yml/badge.svg)](https://github.com/Hai-Wenxiang/StarryVector/actions/workflows/ci.yml)

[English](README.md) | **简体中文**

面向 RAG（检索增强生成）的高性能**向量数据库**，仅支持 **Linux**，使用可移植的 **C++17** 编写（GCC 7+、Clang 5+）。不支持 Windows 平台。

StarryVector 为高维 embedding 向量提供高效的存储、索引与相似度检索，定位是 RAG 流水线中的检索底座。

## 为什么选 StarryVector

- **CPU 上也快** —— AVX2 距离内核运行时分发（带回退），cosine 检索通过入库预归一化归结为内积，HNSW 图索引在 99.5% 召回下比暴力扫描快 6 倍以上。
- **一个二进制跑所有 x86 机器** —— SIMD 通过 `__builtin_cpu_supports` 运行时探测，一次编译从老服务器到新桌面都能跑。设置 `STARRY_FORCE_SCALAR=1` 可强制标量路径。
- **真删除** —— 基于 tombstone 的软删除在两种索引下都生效，删除的行立即从检索结果中消失。
- **单文件持久化** —— `save()` / `load()`，小端版本化格式（`STARRYV2` 存储 HNSW 图；旧的 `STARRYV1` 平面文件仍可加载）。
- **零依赖** —— 核心库只依赖 C++ 标准库。唯一的内置头文件（doctest）仅是开发期测试依赖。
- **API 简洁** —— 状态码代替异常；全程 RAII。

## 功能

- **精确检索** —— SIMD 距离内核（L2、内积、cosine）的暴力 kNN，构造上 100% 召回。O(n log k) 有界堆 top-k 选择。
- **HNSW 近似索引** —— 分层可导航小世界图（Malkov & Yashunin, 2016），支持增量插入、多样性邻居选择、可调 `ef` 搜索束宽。固定随机种子，索引可复现。
- **持久化** —— 单文件 `save()` / `load()`，支持 tombstone 软删除。
- **IVF + 量化**（规划中）—— 第二引擎，内存友好的扫描方式，以精确核心为基准验证。
- **元数据过滤**（规划中）—— 向量检索与标量属性过滤组合。

## 路线图

- [x] 核心向量存储与精确 kNN 检索
- [x] SIMD 距离内核（AVX2 运行时分发，`STARRY_FORCE_SCALAR=1` 可关闭）
- [x] HNSW 索引
- [ ] IVF 索引 + k-means 聚类
- [ ] 元数据过滤
- [ ] WAL + mmap 存储引擎
- [ ] HTTP 服务（REST）接口
- [ ] Python 客户端绑定

## 构建

要求：

- Linux，支持 C++17 的编译器（GCC 7+ 或 Clang 5+）
- CMake 3.10+

```bash
git clone https://github.com/Hai-Wenxiang/StarryVector.git
cd StarryVector
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # 单元测试
```

构建选项：

| 选项 | 默认值 | 说明 |
|---|---|---|
| `STARRY_DISABLE_SIMD` | `OFF` | 完全跳过 AVX2 内核（仅标量） |

## 快速上手

```cpp
#include <starry/db.hpp>

int main() {
    // 精确暴力检索数据库（100% 召回）
    starry::VectorDB db(768);  // embedding 维度

    db.insert(1, {0.1f, 0.2f, /* ... */});   // 返回 starry::kOk
    db.insert(2, {0.3f, 0.4f, /* ... */});

    auto results = db.search({0.1f, 0.2f, /* ... */}, /*k=*/10);
    for (const auto& hit : results) {
        // hit.id, hit.distance —— top-k 最近邻
    }

    db.remove(2);                             // 软删除
    db.save("myvectors.bin");                 // 单文件持久化
}
```

使用 HNSW 近似检索：

```cpp
starry::VectorDB db(768, starry::kL2, starry::kHnswIndex);
db.set_search_ef(128);   // 越大召回越高、越慢

for (/* 每个文档切块 */) {
    db.insert(chunk_id, embedding);
}
auto hits = db.search(query_embedding, 10);   // 近似 top-k
```

## 基准测试与验证

```bash
# 精确检索基准（stdout 输出 JSON 报告）
./build/bin/starry_bench --n 100000 --dim 128 --metric l2 --threads 1

# HNSW 基准（以精确核心为 ground truth 统计召回）
./build/bin/starry_bench --index hnsw --ef 64 --dim 16 --n 100000

# 完整基准矩阵 -> HTML 报告
python3 validation/run_validation.py --open
```

参考数字（单核绑核，i5-13400，n=10万 dim=128，方法学见 `notes/`）：

| 工作负载 | QPS | p50 延迟 | Recall@10 |
|---|---|---|---|
| Flat + AVX2（L2） | ~390 | ~2.5 ms | 100% |
| Flat + AVX2（cosine） | ~390 | ~2.5 ms | 100% |
| HNSW ef=64（dim=16 聚类数据） | ~15,000 | ~60 us | 99.6% |

> 注：召回率取决于数据内在维度。高维**均匀随机**合成数据是图索引的
> 最坏情况；真实 embedding 数据表现要好得多。

详见 [validation/README.md](validation/README.md)。

## 目录结构

```
StarryVector/
├── include/starry/   # 公共头文件（types, distance, flat_index, hnsw_index, db）
├── src/              # 库实现（零依赖）
├── apps/             # 基准测试驱动（starry_bench）
├── examples/         # RAG 检索示例（starry_rag_demo）
├── tests/            # 单元测试（内置 doctest）
├── validation/       # Python harness -> HTML 性能报告
└── third_party/      # 内置的开发期依赖（doctest）
```

## 参与贡献

欢迎提 Issue 和 Pull Request！见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE)
