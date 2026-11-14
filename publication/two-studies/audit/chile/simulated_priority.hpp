#pragma once
// Include after the replay Market definition.
#include <array>

class SimulatedPriority {
    // A separate group for each course and priority category. Edges enter groups
    // in canonical applicant order, independent of the recorded within-tier order.
    vector<array<vector<int>,5>> group;
    int year;
    uint32_t batch;
public:
    SimulatedPriority(const Market& market, const vector<unsigned char>& tier, int yr, uint32_t index)
        : group(market.m), year(yr), batch(index) {
        for(int e=0;e<market.ne;++e) {
            if(tier[e]>4) throw runtime_error("invalid public priority category");
            group[market.prog[e]][tier[e]].push_back(e);
        }
    }
    void draw(vector<int>& out, int repetition) const {
        seed_seq seed{19872137u+batch,static_cast<unsigned>(year),static_cast<unsigned>(repetition)};
        mt19937_64 rng(seed);
        for(const auto& course : group) {
            int position=0;
            for(auto pool : course) {
                // Fisher-Yates with rejection sampling gives an unbiased integer
                // using only the standardized mt19937_64 output bit sequence.
                for(size_t n=pool.size(); n>1; --n) {
                    const uint64_t bound=n, reject=(uint64_t(0)-bound)%bound;
                    uint64_t number;
                    do number=rng(); while(number<reject);
                    swap(pool[n-1],pool[number%bound]);
                }
                for(int edge : pool) out[edge]=position++;
            }
        }
    }
};
