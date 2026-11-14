// Export the private eligibility pools for the second stage after clearing each cohort.
#define main school_replay_entry
#include "replay.cpp"
#undef main
#include "institution.hpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "usage: export_status NUMERIC_PREFIX SIDECAR REGISTER OUTPUT_PREFIX\n";
        return 2;
    }
    const string numeric = argv[1], output = argv[4];
    Market market(numeric + ".bin");
    Institutional rule(market, argv[2]);
    ifstream reg(argv[3], ios::binary);
    const auto school = readv<int>(reg, market.m), april = readv<int>(reg, market.n);
    if (school != rule.school || reg.peek() != EOF) throw runtime_error("register alignment");
    for (const string policy : {"blind", "rank"}) {
        ifstream mask(numeric + "-" + policy + ".mask", ios::binary);
        const auto displayed = readv<char>(mask, market.ne);
        auto outcome = rule.clear(displayed);
        ofstream offers(output + "-" + policy + ".u1", ios::binary);
        ofstream seats(output + "-" + policy + "-remaining.i4", ios::binary);
        ofstream hit(output + "-" + policy + "-hit.u1", ios::binary);
        for (int i = 0; i < market.n; ++i) {
            bool has_april = false;
            for (int e = market.at[i]; e < market.at[i+1]; ++e)
                has_april |= displayed[e] && school[market.prog[e]] == april[i];
            hit.put(has_april ? 1 : 0);
        }
        for (int edge : outcome.partner) offers.put(edge >= 0 ? 1 : 0);
        for (int j = 0; j < market.m; ++j) {
            int32_t remaining = 0;
            for (int category = 0; category < 4; ++category)
                remaining += rule.capacity[4*j+category] - outcome.occupancy[4*j+category];
            if (remaining < 0) throw runtime_error("negative residual seats");
            seats.write(reinterpret_cast<const char*>(&remaining), sizeof(remaining));
        }
        if (!offers || !seats || !hit) throw runtime_error("failed to write private status");
    }
}
