// Paged attention over scheduler-owned CPU KV slots.
// Vector-width dispatch is resolved when compiling for the serving host.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {
constexpr int dim = 128;
constexpr int tile = 64;
constexpr int min_tokens_per_shard = 512;
using bf16 = uint16_t;

inline float unpack(bf16 value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline bf16 pack(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) return bf16((bits >> 16) | 64U);
    return bf16((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
using Vec = __m512;
constexpr int width = 16;
inline Vec zero() { return _mm512_setzero_ps(); }
inline Vec load(const float *p) { return _mm512_loadu_ps(p); }
inline Vec load(const bf16 *p) {
    auto words = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(words), 16));
}
inline Vec broadcast(float x) { return _mm512_set1_ps(x); }
inline Vec muladd(Vec x, Vec y, Vec z) { return _mm512_fmadd_ps(x,y,z); }
inline float sum(Vec x) { return _mm512_reduce_add_ps(x); }
inline void store(float *p, Vec x) { _mm512_storeu_ps(p,x); }
#elif defined(__AVX2__)
using Vec = __m256;
constexpr int width = 8;
inline Vec zero() { return _mm256_setzero_ps(); }
inline Vec load(const float *p) { return _mm256_loadu_ps(p); }
inline Vec load(const bf16 *p) {
    auto words = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(words),16));
}
inline Vec broadcast(float x) { return _mm256_set1_ps(x); }
inline Vec muladd(Vec x, Vec y, Vec z) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(x,y,z);
#else
    return _mm256_add_ps(_mm256_mul_ps(x,y),z);
#endif
}
inline float sum(Vec x) {
    __m128 v = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x,1));
    v = _mm_hadd_ps(v,v);
    return _mm_cvtss_f32(_mm_hadd_ps(v,v));
}
inline void store(float *p, Vec x) { _mm256_storeu_ps(p,x); }
#else
using Vec = float;
constexpr int width = 1;
inline Vec zero() { return 0.0f; }
inline Vec load(const float *p) { return *p; }
inline Vec load(const bf16 *p) { return unpack(*p); }
inline Vec broadcast(float x) { return x; }
inline Vec muladd(Vec x, Vec y, Vec z) { return x*y+z; }
inline float sum(Vec x) { return x; }
inline void store(float *p, Vec x) { *p=x; }
#endif

template<int groups>
struct alignas(64) RowState {
    float accum[groups][dim];
    float max_score[groups], denominator[groups];
};

template<int groups, int state_groups>
void accumulate_row(const bf16 *q, const bf16 *k, const bf16 *v, const int64_t *slots,
                    int length, float scale, RowState<state_groups> &state) {
    static_assert(groups <= state_groups);
    alignas(64) float query[groups][dim], scores[groups][tile];
    auto &accum = state.accum;
    auto &max_score = state.max_score;
    auto &denominator = state.denominator;
    for (int g=0; g<groups; ++g) {
        max_score[g] = -std::numeric_limits<float>::infinity();
        denominator[g] = 0.0f;
        std::fill_n(accum[g], dim, 0.0f);
        for (int j=0; j<dim; ++j) query[g][j] = unpack(q[g*dim+j]);
    }
    for (int begin=0; begin<length; begin+=tile) {
        const int count = std::min(tile, length-begin);
        for (int n=0; n<count; ++n) {
            const bf16 *key = k + slots[begin+n]*dim;
            Vec dots[groups];
            for (int g=0; g<groups; ++g) dots[g]=zero();
            for (int j=0; j<dim; j+=width) {
                const Vec key_vec = load(key+j);
                for (int g=0; g<groups; ++g)
                    dots[g] = muladd(load(query[g]+j), key_vec, dots[g]);
            }
            for (int g=0; g<groups; ++g) scores[g][n]=sum(dots[g])*scale;
        }
        for (int g=0; g<groups; ++g) {
            const float next = std::max(max_score[g],
                    *std::max_element(scores[g], scores[g]+count));
            const float factor = std::exp(max_score[g]-next);
            denominator[g] *= factor;
            for (int j=0; j<dim; ++j) accum[g][j] *= factor;
            max_score[g] = next;
            for (int n=0; n<count; ++n) {
                scores[g][n] = std::exp(scores[g][n]-next);
                denominator[g] += scores[g][n];
            }
        }
        for (int n=0; n<count; ++n) {
            const bf16 *value = v + slots[begin+n]*dim;
            Vec weights[groups];
            for (int g=0; g<groups; ++g) weights[g]=broadcast(scores[g][n]);
            for (int j=0; j<dim; j+=width) {
                const Vec val = load(value+j);
                for (int g=0; g<groups; ++g)
                    store(accum[g]+j, muladd(val, weights[g], load(accum[g]+j)));
            }
        }
    }
}

template<typename Output>
inline Output encode_output(float value) {
    if constexpr (std::is_same_v<Output, float>) return value;
    else return pack(value);
}

template<int groups, typename Output>
void write_row(const RowState<groups> &state, Output *out, float *lse,
               int first = 0, int count = groups) {
    for (int i=0; i<count; ++i) {
        const int g = first + i;
        const float denominator = state.denominator[g];
        const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (int j=0; j<dim; ++j) out[i*dim+j] = encode_output<Output>(state.accum[g][j]*inv);
        if (lse) lse[i] = denominator > 0.0f ? state.max_score[g] + std::log(denominator)
                                             : -std::numeric_limits<float>::infinity();
    }
}

template<int groups, typename Output>
void row(const bf16 *q, const bf16 *k, const bf16 *v, const int64_t *slots,
         int length, float scale, Output *out, float *lse) {
    RowState<groups> state;
    accumulate_row<groups>(q, k, v, slots, length, scale, state);
    write_row(state, out, lse);
}

template<int groups, typename Output>
void write_shards(const RowState<groups> *parts, int shards, Output *out, float *lse,
                  int first = 0, int count = groups) {
    for (int i=0; i<count; ++i) {
        const int g = first + i;
        float max_score = parts[0].max_score[g];
        for (int s=1; s<shards; ++s)
            max_score = std::max(max_score, parts[s].max_score[g]);
        alignas(64) float accum[dim]{};
        float denominator = 0.0f;
        for (int s=0; s<shards; ++s) {
            const auto &part = parts[s];
            const float factor = std::exp(part.max_score[g] - max_score);
            denominator += part.denominator[g] * factor;
            for (int j=0; j<dim; ++j) accum[j] += part.accum[g][j] * factor;
        }
        const float inv = 1.0f / denominator;
        for (int j=0; j<dim; ++j) out[i*dim+j] = encode_output<Output>(accum[j] * inv);
        if (lse) lse[i] = max_score + std::log(denominator);
    }
}

template<int groups, typename Output>
void decode_sharded(const bf16 *q, const bf16 *k, const bf16 *v, const int64_t *slots,
                    const int64_t *lengths, Output *out, float *lse, int jobs, int kv_heads,
                    int stride, int capacity, float scale, int threads, int shards,
                    const int *active_rows) {
    // The decode caller owns this scratch. Keeping it across calls also avoids
    // repeated allocations as consecutive layers reuse the same slot table.
    thread_local std::vector<RowState<groups>> parts;
    parts.resize(jobs * shards);
    auto *partial = parts.data();
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (int i=0; i<jobs*shards; ++i) {
            const int job = i / shards, shard = i % shards;
            const int row = job / kv_heads, h = job % kv_heads;
            const int b = active_rows ? active_rows[row] : row;
            const int64_t begin = lengths[b] * shard / shards;
            const int64_t end = lengths[b] * (shard + 1) / shards;
            const int64_t head_offset = int64_t(h) * capacity * dim;
            const int64_t query_offset = (int64_t(b)*kv_heads + h)*groups*dim;
            accumulate_row<groups>(q + query_offset, k + head_offset,
                v + head_offset, slots + int64_t(b)*stride + begin,
                int(end-begin), scale, partial[i]);
        }
        #pragma omp for schedule(static)
        for (int job=0; job<jobs; ++job) {
            const int row = job / kv_heads, h = job % kv_heads;
            const int b = active_rows ? active_rows[row] : row;
            const int64_t target_head = (int64_t(b)*kv_heads + h)*groups;
            write_shards(partial + job*shards, shards, out + target_head*dim,
                         lse ? lse + target_head : nullptr);
        }
    }
}

template<typename Output>
int decode(const uint16_t *q, const uint16_t *k,
        const uint16_t *v, const int64_t *slots, const int64_t *lengths, Output *out, float *lse,
        int batch, int q_heads, int kv_heads, int stride, int columns, int capacity,
        float scale, int threads) {
    if (kv_heads < 1 || q_heads % kv_heads != 0) return 1;
    const int groups = q_heads / kv_heads;
    if (groups != 1 && groups != 2 && groups != 4 && groups != 8) return 1;
    if (batch <= 0 || threads <= 0) return 2;
    int min_length = columns;
    std::vector<int> active_rows;
    if (lse) active_rows.reserve(batch);
    // Check the actual prefix only. Outside the live prefix the page table may contain padding.
    for (int b=0; b<batch; ++b) {
        if (lengths[b] < (lse ? 0 : 1) || lengths[b] > columns) return 2;
        if (lengths[b] == 0) {
            std::fill_n(out + int64_t(b)*q_heads*dim, q_heads*dim, Output{});
            std::fill_n(lse + int64_t(b)*q_heads, q_heads, -std::numeric_limits<float>::infinity());
            continue;
        }
        if (lse) active_rows.push_back(b);
        min_length = std::min(min_length, int(lengths[b]));
        for (int n=0; n<lengths[b]; ++n) {
            const auto slot = slots[int64_t(b)*stride+n];
            if (slot < 0 || slot >= capacity) return 3;
        }
    }
    const int rows = lse ? int(active_rows.size()) : batch;
    if (!rows) return 0;
    const int jobs = rows*kv_heads;
    const int *row_map = lse ? active_rows.data() : nullptr;
    // A uniform shard count yields equal work per worker on equal-length rows.
    // When the budget cannot fit two complete waves, retain the head split.
    const int shards = std::min(threads / jobs, min_length / min_tokens_per_shard);
    if (shards > 1) {
        threads = jobs * shards;
        switch (groups) {
            case 1: decode_sharded<1>(q,k,v,slots,lengths,out,lse,jobs,kv_heads,stride,capacity,scale,threads,shards,row_map); break;
            case 2: decode_sharded<2>(q,k,v,slots,lengths,out,lse,jobs,kv_heads,stride,capacity,scale,threads,shards,row_map); break;
            case 4: decode_sharded<4>(q,k,v,slots,lengths,out,lse,jobs,kv_heads,stride,capacity,scale,threads,shards,row_map); break;
            case 8: decode_sharded<8>(q,k,v,slots,lengths,out,lse,jobs,kv_heads,stride,capacity,scale,threads,shards,row_map); break;
        }
        return 0;
    }
    threads = std::min(threads, jobs);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int job=0; job<jobs; ++job) {
        const int row_index = job/kv_heads, h=job%kv_heads;
        const int b = row_map ? row_map[row_index] : row_index;
        const int64_t head_offset = (int64_t(b)*kv_heads + h)*groups;
        const auto *query = q + head_offset*dim;
        auto *target = out + head_offset*dim;
        auto *logsumexp = lse ? lse + head_offset : nullptr;
        const auto *ks = k+int64_t(h)*capacity*dim, *vs=v+int64_t(h)*capacity*dim;
        const int len = int(lengths[b]);
        const auto *table = len ? slots+int64_t(b)*stride : nullptr;
        switch (groups) {
            case 1: row<1>(query,ks,vs,table,len,scale,target,logsumexp); break;
            case 2: row<2>(query,ks,vs,table,len,scale,target,logsumexp); break;
            case 4: row<4>(query,ks,vs,table,len,scale,target,logsumexp); break;
            case 8: row<8>(query,ks,vs,table,len,scale,target,logsumexp); break;
        }
    }
    return 0;
}
struct QueryPair {
    int request, first, count;
};

inline bool balance_enabled() {
    const char *flag = std::getenv("SPECSTREAM_CPU_ATTN_BALANCE");
    if (flag) return std::strcmp(flag, "1") == 0;
    // The candidate checkout opts into V2 whenever its grouped path is enabled.
    // Set BALANCE=0 to measure the fixed-size V1 tasks from this same binary.
    const char *grouped = std::getenv("SPECSTREAM_GROUPED_CPU_ATTN");
    return grouped && std::strcmp(grouped, "1") == 0;
}

struct HistoryShard {
    int job, begin, end, partial, work;
};

struct BalancedPlan {
    std::vector<HistoryShard> tasks;
    std::vector<int> offsets;

    bool prepare(const std::vector<QueryPair> &pairs, const int64_t *lengths,
                 int heads, int threads, int uniform_shards) {
        // Estimate time of the existing static OpenMP assignment in tokens per
        // query-head. Avoid paying for partial states and reduction if balanced.
        const int jobs = int(pairs.size()) * heads;
        const int old_tasks = jobs * uniform_shards;
        const int workers = std::min(old_tasks, threads);
        if (threads < 2) return false;
        int64_t total = 0, worst = 0;
        int task = 0;
        for (int worker=0; worker<workers; ++worker) {
            const int count = old_tasks/workers + (worker < old_tasks%workers);
            int64_t load = 0;
            for (int i=0; i<count; ++i, ++task) {
                const auto &p = pairs[task / uniform_shards / heads];
                const auto length = lengths[p.request];
                const int shard = task % uniform_shards;
                load += p.count * (length * (shard+1)/uniform_shards - length*shard/uniform_shards);
            }
            total += load;
            worst = std::max(worst, load);
        }
        // Require enough work AND at least 25% excess on the busiest thread.
        if (total < int64_t(threads)*512 || worst*threads*4 <= total*5) return false;
        const int64_t chunk_work = std::clamp<int64_t>(total / (threads*4), 512, 2048);
        offsets.resize(jobs+1);
        for (int job=0; job<jobs; ++job) {
            const auto &p = pairs[job/heads];
            const int length = int(lengths[p.request]);
            const int chunk = std::max(256, int(chunk_work)/p.count);
            const int shards = 1 + (length-1)/chunk;
            offsets[job] = int(tasks.size());
            for (int s=0; s<shards; ++s) {
                const int begin = int(int64_t(length)*s/shards);
                const int end = int(int64_t(length)*(s+1)/shards);
                tasks.push_back({job,begin,end,int(tasks.size()),(end-begin)*p.count});
            }
        }
        offsets[jobs] = int(tasks.size());
        // Start expensive shards first; their output positions retain the
        // original history order for deterministic FP32 combination.
        std::sort(tasks.begin(), tasks.end(), [](const auto &a, const auto &b) {
            return a.work != b.work ? a.work > b.work : a.partial < b.partial;
        });
        return true;
    }
};

template<int groups, bool balanced=false>
void decode_pairs(const bf16 *q, const bf16 *k, const bf16 *v, const int64_t *slots,
                  const int64_t *lengths, float *out, float *lse,
                  const std::vector<QueryPair> &pairs, int kv_heads, int stride,
                  int capacity, float scale, int threads, int shards,
                  const BalancedPlan *plan=nullptr) {
    // Two time steps and their GQA heads share the same old K and V vectors.
    // The masks for any tokens created in the current round stay on the GPU.
    const int jobs = int(pairs.size()) * kv_heads;
    const int query_stride = kv_heads * groups * dim;
    thread_local std::vector<RowState<groups * 2>> scratch;
    const int tasks = balanced ? int(plan->tasks.size()) : jobs * shards;
    if (balanced || shards > 1) scratch.resize(tasks);
    auto *parts = scratch.data();
    threads = std::min(threads, tasks);
    auto compute = [&](int task) {
            const int job = balanced ? plan->tasks[task].job : task / shards;
            const int head = job % kv_heads;
            const auto &pair = pairs[job / kv_heads];
            const auto head_offset = (int64_t(pair.first)*kv_heads + head)*groups;
            const auto *query = q + head_offset*dim;
            const int64_t length = lengths[pair.request];
            const int64_t begin = balanced ? plan->tasks[task].begin : length * (task%shards) / shards;
            const int64_t end = balanced ? plan->tasks[task].end : length * (task%shards + 1) / shards;
            const auto *table = slots + int64_t(pair.request)*stride + begin;
            const auto kv_offset = int64_t(head)*capacity*dim;
            RowState<groups * 2> local;
            const bool complete = balanced ? plan->offsets[job+1]-plan->offsets[job]==1 : shards==1;
            auto &state = complete ? local : parts[balanced ? plan->tasks[task].partial : task];
            if (pair.count == 2) {
                alignas(64) bf16 queries[2*groups*dim];
                std::copy_n(query, groups*dim, queries);
                std::copy_n(query + query_stride, groups*dim, queries + groups*dim);
                accumulate_row<groups*2>(queries, k+kv_offset, v+kv_offset, table,
                                         int(end-begin), scale, state);
            } else {
                accumulate_row<groups>(query, k+kv_offset, v+kv_offset, table,
                                       int(end-begin), scale, state);
            }
            if (complete) {
                for (int i=0; i<pair.count; ++i) {
                    const auto dest = head_offset + i*kv_heads*groups;
                    write_row(state, out+dest*dim, lse+dest, i*groups, groups);
                }
            }
    };
    #pragma omp parallel num_threads(threads)
    {
        if constexpr (balanced) {
            #pragma omp for schedule(dynamic,1)
            for (int task=0; task<tasks; ++task) compute(task);
        } else {
            #pragma omp for schedule(static)
            for (int task=0; task<tasks; ++task) compute(task);
        }
        if (balanced || shards > 1) {
            #pragma omp for schedule(static)
            for (int job=0; job<jobs; ++job) {
                const auto &pair = pairs[job / kv_heads];
                const int begin = balanced ? plan->offsets[job] : job*shards;
                const int count = balanced ? plan->offsets[job+1]-begin : shards;
                if (count == 1) continue;
                for (int i=0; i<pair.count; ++i) {
                    const auto dest = (int64_t(pair.first+i)*kv_heads + job%kv_heads)*groups;
                    write_shards(parts+begin, count, out+dest*dim, lse+dest,
                                 i*groups, groups);
                }
            }
        }
    }
}
} // namespace

extern "C" int cpu_history_decode(const uint16_t *q, const uint16_t *k,
        const uint16_t *v, const int64_t *slots, const int64_t *lengths, uint16_t *out,
        int batch, int q_heads, int kv_heads, int stride, int columns, int capacity,
        float scale, int threads) {
    return decode(q, k, v, slots, lengths, out, nullptr, batch, q_heads, kv_heads,
                  stride, columns, capacity, scale, threads);
}

extern "C" int cpu_history_decode_grouped_state(
        const bf16 *q, const bf16 *k, const bf16 *v, const int64_t *slots,
        const int64_t *lengths, const int64_t *offsets, float *out, float *lse,
        int requests, int queries, int q_heads, int kv_heads, int stride,
        int columns, int capacity, float scale, int threads) {
    if (requests < 1 || queries < 1 || threads < 1 || offsets[0] != 0 ||
        offsets[requests] != queries) return 2;
    if (kv_heads < 1 || q_heads % kv_heads) return 1;
    const int groups = q_heads / kv_heads;
    if (groups != 1 && groups != 2 && groups != 4 && groups != 8) return 1;
    bool single = requests == queries;
    for (int r=0; r<requests; ++r) {
        if (offsets[r+1] < offsets[r] || offsets[r+1] > queries) return 2;
        single = single && offsets[r+1] == r+1;
    }
    const bool try_balance = balance_enabled();
    if (single && !try_balance)
        return decode(q,k,v,slots,lengths,out,lse,queries,q_heads,kv_heads,stride,
                      columns,capacity,scale,threads);

    std::vector<QueryPair> pairs;
    pairs.reserve(queries);
    int shortest = columns;
    for (int r=0; r<requests; ++r) {
        const auto len = lengths[r];
        if (len < 0 || len > columns) return 2;
        if (!len) {
            const auto heads = (offsets[r+1]-offsets[r])*q_heads;
            std::fill_n(out + offsets[r]*q_heads*dim, heads*dim, 0.f);
            std::fill_n(lse + offsets[r]*q_heads, heads, -std::numeric_limits<float>::infinity());
            continue;
        }
        if (offsets[r+1] == offsets[r]) continue;
        shortest = std::min(shortest, int(len));
        for (int n=0; n<len; ++n) {
            const auto slot = slots[int64_t(r)*stride+n];
            if (slot < 0 || slot >= capacity) return 3;
        }
        for (int64_t i=offsets[r]; i<offsets[r+1]; i+=2)
            pairs.push_back({r, int(i), int(std::min<int64_t>(2,offsets[r+1]-i))});
    }
    if (pairs.empty()) return 0;
    const int jobs = int(pairs.size()) * kv_heads;
    const int shards = std::max(1, std::min(threads/jobs, shortest/min_tokens_per_shard));
    BalancedPlan plan;
    if (try_balance && plan.prepare(pairs, lengths, kv_heads, threads, shards)) {
        switch (groups) {
            case 1: decode_pairs<1,true>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards,&plan); break;
            case 2: decode_pairs<2,true>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards,&plan); break;
            case 4: decode_pairs<4,true>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards,&plan); break;
            case 8: decode_pairs<8,true>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards,&plan); break;
        }
        return 0;
    }
    if (single)
        return decode(q,k,v,slots,lengths,out,lse,queries,q_heads,kv_heads,stride,
                      columns,capacity,scale,threads);
    switch (groups) {
        case 1: decode_pairs<1>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards); break;
        case 2: decode_pairs<2>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards); break;
        case 4: decode_pairs<4>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards); break;
        case 8: decode_pairs<8>(q,k,v,slots,lengths,out,lse,pairs,kv_heads,stride,capacity,scale,threads,shards); break;
    }
    return 0;
}

extern "C" int cpu_history_decode_state(const uint16_t *q, const uint16_t *k,
        const uint16_t *v, const int64_t *slots, const int64_t *lengths, float *out, float *lse,
        int batch, int q_heads, int kv_heads, int stride, int columns, int capacity,
        float scale, int threads) {
    if (balance_enabled() && batch > 0) {
        thread_local std::vector<int64_t> identity;
        if (identity.size() != size_t(batch+1)) {
            identity.resize(size_t(batch)+1);
            for (int i=0; i<=batch; ++i) identity[i] = i;
        }
        return cpu_history_decode_grouped_state(q,k,v,slots,lengths,identity.data(),out,lse,
            batch,batch,q_heads,kv_heads,stride,columns,capacity,scale,threads);
    }
    return decode(q, k, v, slots, lengths, out, lse, batch, q_heads, kv_heads,
                  stride, columns, capacity, scale, threads);
}
