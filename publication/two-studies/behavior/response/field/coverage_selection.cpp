// Reconstruct the annual coverage bridges for each policy before any allocation.
#define main original_replay
#include "replay.cpp"
#undef main
#include "institution.hpp"

struct RecoveryCounts {
    int n = 0, blind_eligible = 0, informed_eligible = 0, eligible_intersection = 0;
    int blind_hit = 0, informed_hit = 0, both_bridges = 0, blind_only = 0, informed_only = 0;
};

vector<char> read_mask(const string& name, int edges) {
    ifstream stream(name, ios::binary);
    auto values = readv<char>(stream, edges);
    if (stream.peek() != EOF) throw runtime_error("wrong mask length");
    return values;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        cerr << "usage: coverage_selection NUMERIC_PREFIX SIDECAR REGISTER YEAR AGGREGATE_CSV\n";
        return 2;
    }
    string prefix = argv[1];
    Market market(prefix+".bin");
    Institutional engine(market, argv[2]);
    array<vector<char>, 2> mask = {read_mask(prefix+"-blind.mask", market.ne), read_mask(prefix+"-rank.mask",market.ne)};
    ifstream reg(argv[3],ios::binary);
    auto school = readv<int>(reg,market.m);
    auto april = readv<int>(reg,market.n);
    if (school != engine.school || reg.peek() != EOF) throw runtime_error("unaligned register");
    vector<int> demand(market.m, 0), capacity(market.m);
    for (int program : market.prog) ++demand[program];
    for (int j = 0; j < market.m; ++j)
        capacity[j]=accumulate(engine.capacity.begin()+4*j,engine.capacity.begin()+4*j+4,0);
    array<RecoveryCounts, 2> by_rule;
    for (int i = 0; i < market.n; ++i) {
        bool original_hit[2] = {}, bridge[2][2] = {}, eligible[2][2] = {};
        for (int policy = 0; policy < 2; ++policy) {
            vector<int> shown;
            for (int e = market.at[i]; e < market.at[i+1]; ++e) {
                if (mask[policy][e]) shown.push_back(school[market.prog[e]]);
            }
            original_hit[policy] = find(shown.begin(),shown.end(),april[i])!=shown.end();
            int first = -1, least = -1;
            for (int e = market.at[i]; e < market.at[i+1]; ++e) {
                int j = market.prog[e];
                if (find(shown.begin(), shown.end(), school[j])!=shown.end()) continue;
                if (first < 0) first = j;
                if (least < 0 || int64_t(demand[j]+1)*(capacity[least]+1) <
                    int64_t(demand[least]+1)*(capacity[j]+1)) least = j;
            }
            for (int rule = 0; rule < 2; ++rule) {
                int j = rule == 0 ? first : least;
                eligible[rule][policy] = j >= 0;
                bridge[rule][policy] = j >= 0 && !original_hit[policy] && april[i] == school[j];
            }
        }
        for (int rule = 0; rule < 2; ++rule) {
            auto& c = by_rule[rule];
            ++c.n;
            c.blind_eligible += eligible[rule][0];
            c.informed_eligible += eligible[rule][1];
            c.eligible_intersection += eligible[rule][0] && eligible[rule][1];
            c.blind_hit += original_hit[0];
            c.informed_hit += original_hit[1];
            bool b = bridge[rule][0], r = bridge[rule][1];
            c.both_bridges += b && r;
            c.blind_only += b && !r;
            c.informed_only += r && !b;
        }
    }
    ofstream out(argv[5]);
    out << "year,scenario,applicants,blind_eligible,informed_eligible,jointly_eligible,blind_original_hits,informed_original_hits,blind_only,common,informed_only\n";
    for (int rule = 0; rule < 2; ++rule) {
        const auto& c = by_rule[rule];
        out << argv[4] << ',' << (rule ? "least_congested_school" : "one_school") << ',' << c.n << ','
            << c.blind_eligible << ',' << c.informed_eligible << ',' << c.eligible_intersection << ','
            << c.blind_hit << ',' << c.informed_hit << ',' << c.blind_only << ',' << c.both_bridges << ','
            << c.informed_only << '\n';
    }
    return out ? 0 : 1;
}
