// Compare the frozen predictors with first-five and uniform menus on the same cohorts.
#define main archived_replay_main
#include "replay.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    Market market(argv[1]);
    int year = stoi(argv[2]);
    ofstream out(argv[3]);
    out << "year,policy,draw,applicants,placed,covered_full,covered_recorded,official_offers\n";
    vector<char> full(market.ne, 1);
    const auto target = market.da(full);
    market.verify(target, full);
    for (int rep = 0; rep <= 48; ++rep) {
        mt19937_64 rng(28311 + year * 1000 + rep);
        vector<uint64_t> tie(market.ne);
        generate(tie.begin(), tie.end(), [&] { return rng(); });
        vector<string> policies = rep ? vector<string>{"uniform"} : vector<string>{"first", "blind", "rank"};
        for (const auto& policy : policies) {
            const auto& score = policy == "rank" ? market.rank : market.blind;
            const auto shown = market.menu(score, tie, 5, policy);
            const auto matches = market.da(shown);
            market.verify(matches, shown);
            int placed = 0, recovered = 0, covered = 0, offers = 0;
            for (int i = 0; i < market.n; ++i) {
                placed += matches[i] >= 0;
                recovered += target[i] >= 0 && shown[target[i]];
                offers += market.official[i] >= 0;
                bool hit = false;
                for (int e = market.at[i]; e < market.at[i+1]; ++e) {
                    if (shown[e] && market.prog[e] == market.official[i]) hit = true;
                }
                covered += hit;
            }
            out << year << ',' << policy << ',' << rep << ',' << market.n << ',' << placed << ','
                << recovered << ',' << covered << ',' << offers << '\n';
        }
    }
    return out ? 0 : 1;
}
