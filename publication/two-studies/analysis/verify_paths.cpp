// Exhaust the strict 3-by-3 unit market (up to two label symmetries).
// Direct blocking-pair enumeration makes no calls to deferred acceptance.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

constexpr int n = 3;
constexpr int graphs = 1 << (n * n);
using Row = std::array<int, n>;
using Ranks = std::array<Row, n>;
using Count = std::uint64_t;

int bit(int applicant, int program) {
    return 1 << (n * applicant + program);
}

struct Matching {
    Row partner;
    Row holder{-1, -1, -1};
    int required = 0;
};

std::vector<Matching> partial_matchings() {
    std::vector<Matching> list;
    for (int a = -1; a < n; ++a) for (int b = -1; b < n; ++b) for (int c = -1; c < n; ++c) {
        Matching m{{a,b,c}};
        bool valid = true;
        for (int i = 0; i < n; ++i) {
            const int j = m.partner[i];
            if (j < 0) continue;
            if (m.holder[j] >= 0) { valid = false; break; }
            m.holder[j] = i;
            m.required |= bit(i,j);
        }
        if (valid) list.push_back(m);
    }
    assert(list.size() == 34);
    return list;
}

int blocks(const Matching& m, const Ranks& pref, const Ranks& priority) {
    int blocking = 0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        const bool applicant_prefers = m.partner[i] < 0 || pref[i][j] < pref[i][m.partner[i]];
        const bool program_prefers = m.holder[j] < 0 || priority[j][i] < priority[j][m.holder[j]];
        if (applicant_prefers && program_prefers) blocking |= bit(i,j);
    }
    return blocking;
}

void check_paths(const Matching& before, const Matching& after, int display, bool one_slot,
                 const Ranks& pref, const Ranks& priority, std::array<Count,n+1>& lengths) {
    int visited = 0;
    for (int start = 0; start < n; ++start) {
        const int j = before.partner[start];
        if (j < 0 || !(display & bit(start, j)) || after.partner[start] >= 0) continue;
        int i = start, length = 0;
        while (display & bit(i,before.partner[i])) {
            assert(!(visited & (1 << i)));
            visited |= (1 << i);
            const int old = before.partner[i];
            const int next = after.holder[old];
            assert(next >= 0 && priority[old][next] < priority[old][i]);
            assert(before.partner[next] >= 0 && pref[next][before.partner[next]] < pref[next][old]);
            i = next;
            ++length;
        }
        assert(i != start && !(visited & (1 << i)) && after.partner[i] >= 0);
        visited |= 1 << i;
        assert(length < n && (!one_slot || length == 1));
        ++lengths[length];
    }
}

int main() {
    const auto matchings = partial_matchings();
    Row id{0,1,2};
    std::vector<Row> orders;
    do { orders.push_back(id); } while (std::next_permutation(id.begin(), id.end()));
    std::array<bool, graphs> one_slot{};
    for (int d = 0; d < graphs; ++d) {
        one_slot[d] = true;
        for (int i = 0; i < n; ++i)
            if (__builtin_popcount((d >> (i*n)) & 7) > 1) one_slot[d] = false;
    }
    Count profiles = 0, outcome_pairs = 0, quota_one_pairs = 0;
    std::array<Count,n+1> lengths{};
    for (auto a1 : orders) for (auto a2 : orders) for (auto s0 : orders)
    for (auto s1 : orders) for (auto s2 : orders) {
        // Rename programs so applicant 0 ranks them by label. Swap applicants 1 and 2
        // if needed so program 0 prefers 1 over 2. Orders here map partners to ranks.
        if (s0[1] > s0[2]) continue;
        ++profiles;
        Ranks pref{id,a1,a2}, priority{s0,s1,s2};
        std::array<std::vector<int>, graphs> stable;
        for (int k = 0; k < int(matchings.size()); ++k) {
            const auto& m = matchings[k];
            const int blocking = blocks(m,pref,priority);
            for (int d = 0; d < graphs; ++d)
                if ((d & m.required) == m.required && !(d & blocking)) stable[d].push_back(k);
        }
        for (int e = 0; e < graphs; ++e) {
            for (int d = e;; d = (d-1) & e) {
                for (int i : stable[e]) for (int j : stable[d]) {
                    ++outcome_pairs;
                    quota_one_pairs += one_slot[d];
                    check_paths(matchings[i],matchings[j],d,one_slot[d],pref,priority,lengths);
                }
                if (d == 0) break;
            }
        }
    }
    std::cout << "{\"symmetry_reduced_profiles\":" << profiles
              << ",\"stable_outcome_pairs\":" << outcome_pairs
              << ",\"pairs_at_most_one_displayed_program_per_applicant\":" << quota_one_pairs
              << ",\"covered_unmatched_applicant_paths_of_length_1\":" << lengths[1]
              << ",\"covered_unmatched_applicant_paths_of_length_2\":" << lengths[2] << "}\n";
}
