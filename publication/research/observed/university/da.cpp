// Applicant-proposing deferred acceptance with responsive program capacities.
// Edges are stored in submitted order per student; smaller priority values win.
#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>
extern "C" int solve(int n, int m, const int* offset, const int* school,
                     const int* priority, const int* cap, const uint8_t* rank,
                     unsigned limit, int* out) {
    using Entry=std::pair<int,int>; // priority and proposing student
    std::vector<std::priority_queue<Entry>> held(m);
    std::vector<int> next(offset,offset+n), free;
    free.reserve(n);
    std::fill(out,out+n,-1);
    for (int i=n-1;i>=0;--i) free.push_back(i);
    int proposals=0;
    while (!free.empty()) {
        int i=free.back();free.pop_back();
        while (next[i]<offset[i+1]) {
            int e=next[i]++;
            if (rank[e]>=limit) continue;
            ++proposals;
            int j=school[e];
            if (cap[j]==0) continue;
            auto& heap=held[j];
            if (static_cast<int>(heap.size())==cap[j]) {
                if (heap.top().first < priority[e]) continue;
                int displaced=heap.top().second;
                heap.pop();out[displaced]=-1;free.push_back(displaced);
            }
            heap.emplace(priority[e],i);out[i]=e;break;
        }
    }
    return proposals;
}
