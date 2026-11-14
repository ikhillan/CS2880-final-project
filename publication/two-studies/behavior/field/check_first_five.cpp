// Independent serial DA on first-five reports, used to check the grouped placement counts.
#define main original_replay_main
#include "replay.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    Market market(argv[1]);
    std::vector<uint64_t> unused_keys(market.ne);
    const auto shown = market.menu(market.blind, unused_keys, 5, "first");
    const auto chosen = market.da(shown);
    market.verify(chosen, shown);
    std::vector<int32_t> programs(market.n, -1);
    for (int i = 0; i < market.n; ++i) {
        if (chosen[i] >= 0) programs[i] = market.prog[chosen[i]];
    }
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(programs.data()), programs.size() * sizeof(int32_t));
    return out ? 0 : 1;
}
