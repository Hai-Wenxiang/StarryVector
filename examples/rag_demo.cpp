// StarryVector - minimal end-to-end RAG retrieval demo.
//
// This example shows exactly where a vector database sits inside a RAG
// (Retrieval-Augmented Generation) pipeline:
//
//   documents --> chunk --> embed --> vectors --insert--> StarryVector
//                                                            |
//   question --> embed --> query vector --search(top-k)---> ids
//                                                            |
//                                            sidecar store (id -> passage)
//                                                            |
//                                       retrieved passages + question
//                                                     |
//                                          assembled LLM prompt (printed)
//
// Three things RAG needs from the store - batch insert (offline),
// top-k search (online) and delete (when documents change) - are all
// covered by the VectorDB API.  What the store NEVER holds is the text
// itself: the library returns ids, the passage text lives in a sidecar
// store owned by the application (a plain array here; SQLite, a JSON
// file or your business DB in production).
//
// To stay dependency-free, the embedding function below is a toy:
// a bag-of-tokens hash embedding (term frequency, L2-normalised,
// compared with cosine).  It retrieves by keyword overlap well enough
// to demonstrate the wiring.  In a production system you replace
// `embed()` with a real embedding model (BGE / GTE / text-embedding-*)
// running OUTSIDE the library - the interface (insert dim floats,
// search with the same dim floats) does not change at all.
//
// Build & run:
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
//   ./build/bin/starry_rag_demo
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "starry/db.hpp"

namespace {

// Embedding dimension.  Purely internal to this demo - it only needs to
// be fixed so that passages and queries land in the same space.  Hash
// bucket collisions add noise; 512 buckets is plenty for this corpus.
const std::size_t kDim = 512;

const char* kDbPath = "/tmp/starry_rag_demo.bin";

// ---------------------------------------------------------------------------
// Toy tokenizer: latin/digit runs -> lowercase words; CJK runs -> character
// bigrams ("向量库" -> 向量, 量库), the simplest scheme for a language
// without spaces.  Everything else separates tokens.
// ---------------------------------------------------------------------------

bool is_cjk(std::uint32_t cp) {
  return cp >= 0x4E00u && cp <= 0x9FFFu;  // CJK Unified Ideographs
}

// Encodes one code point as UTF-8 (tokens are stored back as strings so
// that the hash function sees stable byte sequences).
std::string utf8(std::uint32_t cp) {
  std::string s;
  if (cp < 0x80u) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800u) {
    s.push_back(static_cast<char>(0xC0u | (cp >> 6)));
    s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else if (cp < 0x10000u) {
    s.push_back(static_cast<char>(0xE0u | (cp >> 12)));
    s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else {
    s.push_back(static_cast<char>(0xF0u | (cp >> 18)));
    s.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
    s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  }
  return s;
}

std::vector<std::string> tokenize(const std::string& text) {
  std::vector<std::string> tokens;
  std::string word;                // pending latin/digit word
  std::vector<std::uint32_t> cjk;  // pending CJK run

  // Lambdas kept tiny; flushing a pending word/run is needed whenever a
  // different character class begins.
  // (capturing by reference is fine, all locals outlive the calls)
  struct Flusher {
    std::vector<std::string>& tokens;
    std::string& word;
    std::vector<std::uint32_t>& cjk;
    void word_() {
      if (!word.empty()) {
        tokens.push_back(word);
        word.clear();
      }
    }
    void cjk_() {
      if (cjk.empty()) return;
      if (cjk.size() == 1) {
        tokens.push_back(utf8(cjk[0]));
      } else {
        for (std::size_t i = 0; i + 1 < cjk.size(); ++i) {
          tokens.push_back(utf8(cjk[i]) + utf8(cjk[i + 1]));
        }
      }
      cjk.clear();
    }
  } flush = {tokens, word, cjk};

  for (std::size_t i = 0; i < text.size();) {
    const unsigned char b = static_cast<unsigned char>(text[i]);
    std::uint32_t cp;
    int len;
    if (b < 0x80u) {
      cp = b;
      len = 1;
    } else if ((b & 0xE0u) == 0xC0u) {
      cp = b & 0x1Fu;
      len = 2;
    } else if ((b & 0xF0u) == 0xE0u) {
      cp = b & 0x0Fu;
      len = 3;
    } else {
      cp = b & 0x07u;
      len = 4;
    }
    bool ok = true;
    for (int j = 1; j < len; ++j) {
      if (i + static_cast<std::size_t>(j) >= text.size()) {
        ok = false;  // truncated sequence: treat as separator
        break;
      }
      const unsigned char cb =
          static_cast<unsigned char>(text[i + static_cast<std::size_t>(j)]);
      if ((cb & 0xC0u) != 0x80u) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (cb & 0x3Fu);
    }
    if (!ok) {
      flush.word_();
      flush.cjk_();
      ++i;
      continue;
    }

    const bool ascii_alnum =
        (len == 1) && (std::isalnum(static_cast<int>(b)) != 0);
    if (ascii_alnum) {
      flush.cjk_();
      word += static_cast<char>(std::tolower(static_cast<int>(b)));
    } else if (is_cjk(cp)) {
      flush.word_();
      cjk.push_back(cp);
    } else {
      flush.word_();
      flush.cjk_();
    }
    i += static_cast<std::size_t>(len);
  }
  flush.word_();
  flush.cjk_();
  return tokens;
}

// FNV-1a 64-bit over the token bytes; deterministic across runs and
// platforms, which is all an embedding hash needs to be.
std::uint64_t fnv1a(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < s.size(); ++i) {
    h ^= static_cast<unsigned char>(s[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

// Toy embedding: each token votes (+1) for its hash bucket, then the
// vector is L2-normalised so cosine degenerates to a dot product.
// Replace this function with a real embedding model in production.
std::vector<float> embed(const std::string& text) {
  std::vector<float> v(kDim, 0.0f);
  const std::vector<std::string> tokens = tokenize(text);
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    v[static_cast<std::size_t>(fnv1a(tokens[t]) % kDim)] += 1.0f;
  }
  float norm = 0.0f;
  for (std::size_t i = 0; i < kDim; ++i) {
    norm += v[i] * v[i];
  }
  norm = std::sqrt(norm);
  if (norm > 0.0f) {
    for (std::size_t i = 0; i < kDim; ++i) {
      v[i] /= norm;
    }
  }
  return v;
}

// ---------------------------------------------------------------------------
// The corpus: this stands for "your documents after chunking".  The id
// assigned to each passage is simply its index; the mapping id -> text
// is the sidecar store RAG applications own next to the vector library.
// ---------------------------------------------------------------------------

struct Passage {
  const char* doc;
  const char* text;
};

std::vector<Passage> build_corpus() {
  std::vector<Passage> passages;
  passages.push_back(Passage{"rag.md",
    "RAG（检索增强生成）先从知识库中检索与问题相关的段落，再把段落和问题一起交给大语言模型生成回答，从而让模型能够利用私有知识。"});
  passages.push_back(Passage{"rag.md",
    "RAG 系统的核心指标是检索质量：如果检索不到正确的段落，再强的生成模型也会答错，而检索这一步通常由向量数据库完成。"});
  passages.push_back(Passage{"vectordb.md",
    "向量数据库为每个文本块存储一个嵌入向量，并支持按向量相似度检索最相近的 k 个结果，它是 RAG 检索环节的存储引擎。"});
  passages.push_back(Passage{"vectordb.md",
    "StarryVector 是一个纯 C++ 实现的向量库，提供精确检索与近似检索两种模式，并支持单文件持久化。"});
  passages.push_back(Passage{"chunking.md",
    "文档切分常见策略有固定长度切分、按句子切分和按语义切分，块太小会丢失上下文，太大会稀释语义。"});
  passages.push_back(Passage{"chunking.md",
    "经验上 RAG 的文本块大小常取 200 到 500 个 token，并让相邻块之间保留少量重叠，避免关键信息被切断在两个块里。"});
  passages.push_back(Passage{"embedding.md",
    "嵌入模型把文本映射为高维向量，语义相近的文本在向量空间中距离更近，常见向量维度有 768 和 1024。"});
  passages.push_back(Passage{"embedding.md",
    "中文嵌入模型如 BGE 与 GTE 在检索任务上表现良好，使用时通常比较余弦相似度，向量需要先归一化。"});
  passages.push_back(Passage{"index.md",
    "HNSW 是一种多层图结构的近似最近邻索引，查询时从顶层入口逐层贪心下降，能在亿级向量上毫秒级返回结果。"});
  passages.push_back(Passage{"index.md",
    "IVF 索引先用 k-means 把向量聚成若干簇，查询时只扫描离问题最近的几个簇，以少量精度损失换取大幅加速。"});
  passages.push_back(Passage{"index.md",
    "精确检索逐个计算查询与所有向量的距离，结果完全准确，适合中小规模知识库或作为近似索引的正确性基准。"});
  passages.push_back(Passage{"eval.md",
    "评估 RAG 检索质量常用召回率 recall@k：前 k 条结果命中正确段落的比例，可以用标注好的问答对自动计算。"});
  return passages;
}

// Assembles exactly the string a RAG application would hand to the LLM.
std::string build_prompt(const std::string& question,
                         const std::vector<starry::SearchResult>& hits,
                         const std::vector<Passage>& passages) {
  std::ostringstream prompt;
  prompt << "你是知识库助手，请只依据下面的参考资料回答问题。\n\n"
         << "参考资料：\n";
  for (std::size_t i = 0; i < hits.size(); ++i) {
    // Production note: this lookup would go to the sidecar store
    // (SQL query, cache hit, ...), never into the vector library.
    prompt << "[" << (i + 1) << "] " << passages[hits[i].id].text << "\n";
  }
  prompt << "\n问题：" << question << "\n";
  return prompt.str();
}

// Runs one retrieval round trip: embed -> search -> sidecar lookup.
void run_query(const starry::VectorDB& db,
               const std::vector<Passage>& passages,
               const std::string& question, std::size_t k,
               bool show_prompt) {
  std::cout << "问题: " << question << "\n";
  const std::vector<float> query_vec = embed(question);
  const std::vector<starry::SearchResult> hits = db.search(query_vec, k);
  for (std::size_t i = 0; i < hits.size(); ++i) {
    std::cout << "  [" << (i + 1) << "] dist=" << std::fixed
              << std::setprecision(3) << hits[i].distance << "  ("
              << passages[hits[i].id].doc << ") "
              << passages[hits[i].id].text << "\n";
  }
  std::cout.unsetf(std::ios::fixed);
  if (show_prompt) {
    std::cout << "\n----- 拼装好的 LLM prompt（真实系统在此调用大模型） -----\n"
              << build_prompt(question, hits, passages)
              << "--------------------------------------------------------\n";
  }
  std::cout << "\n";
}

}  // namespace

int main() {
  std::cout << "=== StarryVector RAG 检索示例 ===\n\n";

  // ---- 1. ingest: chunk -> embed -> insert --------------------------------
  const std::vector<Passage> passages = build_corpus();
  starry::VectorDB db(kDim, starry::kCosine);
  for (std::size_t i = 0; i < passages.size(); ++i) {
    const std::vector<float> vec = embed(passages[i].text);
    if (db.insert(static_cast<starry::id_t>(i), vec) != starry::kOk) {
      std::cerr << "insert failed for passage " << i << "\n";
      return 1;
    }
  }
  std::cout << "ingested " << db.size() << " passages (dim=" << db.dim()
            << ", metric=cosine)\n\n";

  // ---- 2. persist + reopen -------------------------------------------------
  // Demonstrates the serving flow: an offline indexing process saves the
  // file, a later (possibly separate) query process loads it.
  if (db.save(kDbPath) != starry::kOk) {
    std::cerr << "save failed\n";
    return 1;
  }
  starry::Status status = starry::kOk;
  std::unique_ptr<starry::VectorDB> served = starry::VectorDB::load(kDbPath,
                                                                    &status);
  if (!served) {
    std::cerr << "load failed, status=" << status << "\n";
    return 1;
  }
  std::cout << "persisted to " << kDbPath << " and reloaded ("
            << served->size() << " passages alive)\n\n";

  // ---- 3. retrieval --------------------------------------------------------
  run_query(*served, passages, "RAG 是什么？", 3, /*show_prompt=*/true);
  run_query(*served, passages, "HNSW 索引是怎么工作的？", 3, false);
  run_query(*served, passages, "文档应该怎么切分成文本块？", 3, false);

  // ---- 4. update: a document was edited -> delete its chunk ----------------
  const starry::id_t stale_chunk = 4;  // first chunking.md passage
  std::cout << "删除过期的块 id=" << stale_chunk << " 后重新检索：\n";
  served->remove(stale_chunk);
  run_query(*served, passages, "文档应该怎么切分成文本块？", 3, false);

  return 0;
}
