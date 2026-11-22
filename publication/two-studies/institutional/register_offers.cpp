// Group register-linked pupils after clearing the complete national market.
#define main school_replay_entry
#include "replay.cpp"
#undef main
#include "institution.hpp"
#include <map>

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        cerr << "usage: register_offers PREFIX SIDECAR REGISTER YEAR BLIND_CSV [FIRST_FIVE_CSV]\n";
        return 2;
    }
    const string prefix = argv[1];
    Market market(prefix + ".bin");
    Institutional institution(market, argv[2]);
    array<vector<char>, 3> menus;
    array<vector<int>, 3> partners;
    for (int k = 0; k < 2; ++k) {
        ifstream mask(prefix + (k ? "-rank.mask" : "-blind.mask"), ios::binary);
        menus[k] = readv<char>(mask, market.ne);
        partners[k] = institution.clear(menus[k]).partner;
    }
    if (argc == 7) {
        vector<uint64_t> unused(market.ne);
        menus[2] = market.menu(market.blind, unused, 5, "first");
        partners[2] = institution.clear(menus[2]).partner;
    }
    ifstream file(argv[3], ios::binary);
    auto schools = readv<int>(file, market.m);
    auto april = readv<int>(file, market.n);
    if (schools != institution.school || file.peek() != EOF) throw runtime_error("register alignment");
    for (int policy : {0, 2}) {
        if (policy == 2 && argc < 7) break;
        map<string, array<int, 8>> groups;
        for (int i = 0; i < market.n; ++i) {
            if (april[i] < 0) continue;
            bool a_hit = false, b_hit = false, changed = false;
            for (int e = market.at[i]; e < market.at[i+1]; ++e) {
                changed |= menus[policy][e] != menus[1][e];
                if (schools[market.prog[e]] == april[i]) {
                    a_hit |= menus[policy][e];
                    b_hit |= menus[1][e];
                }
            }
            bool a = partners[policy][i] >= 0, b = partners[1][i] >= 0;
            int offered = market.official[i];
            const string group = offered < 0 ? "no_initial_offer" :
                (schools[offered] == april[i] ? "enrolled_at_initial_offer" : "enrolled_elsewhere");
            array<int, 8> values{1, a_hit, b_hit, a, b, a && !b, !a && b, changed};
            for (size_t k = 0; k < values.size(); ++k) {
                groups["linked"][k] += values[k];
                groups[group][k] += values[k];
            }
        }
        const string name = policy == 2 ? "first" : "blind";
        ofstream out(argv[policy == 2 ? 6 : 5]);
        out << "year,group,linked," << name << "_hits,rank_hits," << name
            << "_placed,rank_placed,exits,entrants,changed_menu\n";
        for (const auto& [group, row] : groups) {
            out << argv[4] << ',' << group;
            for (int value : row) out << ',' << value;
            out << '\n';
        }
        if (!out) throw runtime_error("failed to write group comparison");
    }
}
