// Reuse the archived DA clearing and within-tier lottery. The driver checks the
// original comparator against its archived results, batch by batch.
#define main archived_replay_main
#include "replay.cpp"
#undef main
#include "simulated_priority.hpp"
#include <array>
#include <tuple>
using Word = uint64_t;
using Choice = array<int, 5>;
using Score = tuple<int, int, int>;
int bits(Word x) { return __builtin_popcountll(x); }

vector<char> cutoff_attainable(const Market &m, const vector<int> &match) {
    vector<int> used(m.m), worst(m.m, -1);
    for (int e : match)
        if (e >= 0) {
            int j = m.prog[e];
            ++used[j];
            worst[j] = max(worst[j], m.priority[e]);
        }
    vector<char> attainable(m.ne);
    for (int i = 0; i < m.n; ++i) {
        int best = -1;
        for (int e = m.at[i]; e < m.at[i + 1]; ++e) {
            int j = m.prog[e];
            attainable[e] = m.cap[j] > 0 && (used[j] < m.cap[j] || m.priority[e] <= worst[j]);
            if (best < 0 && attainable[e])
                best = e;
        }
        if (best != match[i])
            throw logic_error("highest-ranked attainable edge differs from the full DA partner");
    }
    return attainable;
}

struct RowSearch {
    const vector<int> &eligible;
    const vector<int> &hits;
    const vector<Word> &attainable;
    vector<Word> remaining;
    Choice best, selected{};
    Score incumbent;
    RowSearch(const vector<int> &edges, const vector<int> &hits_, const vector<Word> &attainable_,
              Choice start, Word protected_bits)
        : eligible(edges), hits(hits_), attainable(attainable_), remaining(edges.size() + 1),
          best(start) {
        sort(best.begin(), best.end());
        Word covered = protected_bits;
        int named = 0, redundancy = 0;
        for (int e : best) {
            covered |= attainable[e];
            named += hits[e];
            redundancy += bits(attainable[e]);
        }
        incumbent = {bits(covered), named, redundancy};
        for (int p = int(edges.size()) - 1; p >= 0; --p)
            remaining[p] = remaining[p + 1] | attainable[edges[p]];
        // Initial portfolio globally optimizes the secondary criteria; if it meets
        // the full-list coverage bound it already solves the entire problem.
        if (covered != (protected_bits | remaining[0]))
            search(0, 0, protected_bits, 0, 0);
    }
    void search(int pos, int depth, Word covered, int named, int redundancy) {
        if (depth == 5) {
            Score candidate{bits(covered), named, redundancy};
            if (candidate > incumbent || (candidate == incumbent && selected < best)) {
                best = selected;
                incumbent = candidate;
            }
            return;
        }
        if (bits(covered | remaining[pos]) < get<0>(incumbent))
            return;
        for (int p = pos; p <= int(eligible.size()) - (5 - depth); ++p) {
            int e = eligible[p];
            selected[depth] = e;
            search(p + 1, depth + 1, covered | attainable[e], named + hits[e],
                   redundancy + bits(attainable[e]));
        }
    }
};

Choice backup_tie_portfolio(vector<int> edges, const vector<int> &hits,
                            const vector<Word> &attainable) {
    stable_sort(edges.begin(), edges.end(), [&](int a, int b) {
        return make_pair(hits[a], bits(attainable[a])) > make_pair(hits[b], bits(attainable[b]));
    });
    Choice selected;
    copy_n(edges.begin(), 5, selected.begin());
    return selected;
}

// Expose just the discrete optimizer to an independently exhaustive checker.
int solve_test_rows() {
    int n;
    Word protected_bits;
    while (cin >> n >> protected_bits) {
        if (n < 5)
            throw runtime_error("oracle comparison needs at least five options");
        vector<int> edges(n), hits(n);
        vector<Word> attainable(n);
        iota(edges.begin(), edges.end(), 0);
        for (int i = 0; i < n; ++i)
            cin >> hits[i] >> attainable[i];
        auto initial = backup_tie_portfolio(edges, hits, attainable);
        auto choice = RowSearch(edges, hits, attainable, initial, protected_bits).best;
        for (int edge : choice)
            cout << edge << ' ';
        cout << '\n';
    }
    return 0;
}

struct Menus {
    vector<char> mc, ties, coverage;
    Menus(const Market &m, const vector<int> &hits, const vector<Word> &attainable) {
        mc.assign(m.ne, 1);
        ties = mc;
        coverage = mc;
        vector<int> eligible, sorted;
        for (int i = 0; i < m.n; ++i) {
            if (m.degree[i] <= 5)
                continue;
            eligible.clear();
            Word protected_bits = 0;
            for (int e = m.at[i]; e < m.at[i + 1]; ++e) {
                if (m.prog[e] == m.ret[i])
                    protected_bits |= attainable[e];
                else {
                    eligible.push_back(e);
                    mc[e] = ties[e] = coverage[e] = 0;
                }
            }
            sorted = eligible;
            stable_sort(sorted.begin(), sorted.end(),
                        [&](int a, int b) { return hits[a] > hits[b]; });
            for (int k = 0; k < 5; ++k)
                mc[sorted[k]] = 1;
            auto initial = backup_tie_portfolio(eligible, hits, attainable);
            for (int e : initial)
                ties[e] = 1;
            RowSearch solver(eligible, hits, attainable, initial, protected_bits);
            for (int e : solver.best)
                coverage[e] = 1;
        }
    }
};

void evaluate(ostream &out, int year, int batch, string name, const Market &m,
              const vector<char> &menu, const vector<char> &mc, const vector<char> &tie,
              const vector<int> &mc_match, const vector<int> &full,
              const vector<char> &realized_att, const vector<int> &hits,
              const vector<Word> &attainable, const vector<int> &rbd, const vector<int> &april) {
    auto result = m.da(menu);
    m.verify(result, menu);
    int placed = 0, initial = 0, fullshown = 0, named = 0, first = 0, linked = 0, linkedplaced = 0,
        aprilmatched = 0, nooffer = 0, noofferplaced = 0, noofferlinked = 0, noofferapril = 0,
        gain = 0, loss = 0, nooffergain = 0, noofferlost = 0, changed = 0, changedtie = 0,
        indirectgain = 0, indirectloss = 0, releasedcov = 0;
    long long traincov = 0, trainhit = 0, trainredundancy = 0;
    int aprilshown = 0, noofferaprilshown = 0;
    for (int i = 0; i < m.n; ++i) {
        int e = result[i];
        bool placed_i = e >= 0, linked_i = april[i] >= 0, absent = m.official[i] < 0;
        placed += placed_i;
        fullshown += full[i] >= 0 && menu[full[i]];
        named += full[i] >= 0 && e == full[i];
        first += placed_i && m.at[i] == e && m.prog[e] != m.appended[i];
        bool enroll = placed_i && linked_i && rbd[m.prog[e]] == april[i];
        linked += linked_i;
        linkedplaced += linked_i && placed_i;
        aprilmatched += enroll;
        nooffer += absent;
        noofferplaced += absent && placed_i;
        noofferlinked += absent && linked_i;
        noofferapril += absent && enroll;
        bool gained = placed_i && mc_match[i] < 0, lost = !placed_i && mc_match[i] >= 0;
        gain += gained;
        loss += lost;
        nooffergain += absent && gained;
        noofferlost += absent && lost;
        Word unionbits = 0;
        bool diffmc = false, diffties = false, releasecov = false, aprilshown_i = false;
        int visible = 0;
        for (int t = m.at[i]; t < m.at[i + 1]; ++t) {
            diffmc |= menu[t] != mc[t];
            diffties |= menu[t] != tie[t];
            if (menu[t]) {
                ++visible;
                unionbits |= attainable[t];
                trainhit += hits[t];
                trainredundancy += bits(attainable[t]);
                releasecov |= realized_att[t];
                initial += m.prog[t] == m.official[i];
                aprilshown_i |= linked_i && rbd[m.prog[t]] == april[i];
            }
        }
        aprilshown += aprilshown_i;
        noofferaprilshown += absent && aprilshown_i;
        if (visible != min(5, m.degree[i]) + (m.ret[i] >= 0))
            throw logic_error("wrong shortlist size");
        traincov += bits(unionbits);
        releasedcov += releasecov;
        changed += diffmc;
        changedtie += diffties;
        indirectgain += !diffmc && gained;
        indirectloss += !diffmc && lost;
    }
    out << year << ',' << batch << ',' << name << ',' << m.n << ',' << placed << ',' << initial
        << ',' << fullshown << ',' << named << ',' << first << ',' << linked << ',' << linkedplaced
        << ',' << aprilmatched << ',' << nooffer << ',' << noofferplaced << ',' << noofferlinked
        << ',' << noofferapril << ',' << gain << ',' << loss << ',' << nooffergain << ','
        << noofferlost << ',' << changed << ',' << changedtie << ',' << indirectgain << ','
        << indirectloss << ',' << releasedcov << ',' << traincov << ',' << trainhit << ','
        << trainredundancy << ',' << aprilshown << ',' << noofferaprilshown << '\n';
}

// One uniform permutation of applicants per scenario, shared across every course.
void common_draw(vector<int> &out, const Market &m, const vector<unsigned char> &tiers, int year,
                 int batch, int draw) {
    seed_seq seed{19872137u + static_cast<unsigned>(batch), static_cast<unsigned>(year),
                  static_cast<unsigned>(draw)};
    mt19937_64 rng(seed);
    vector<int> order(m.n);
    iota(order.begin(), order.end(), 0);
    for (size_t n = order.size(); n > 1; --n) {
        uint64_t bound = n, reject = (uint64_t(0) - bound) % bound;
        uint64_t number;
        do
            number = rng();
        while (number < reject);
        swap(order[n - 1], order[number % bound]);
    }
    for (int position = 0; position < m.n; ++position) {
        int pupil = order[position];
        for (int e = m.at[pupil]; e < m.at[pupil + 1]; ++e)
            out[e] = int(tiers[e]) * m.n + position;
    }
}

int main(int argc, char **argv) {
    if (argc == 2 && string(argv[1]) == "--check-rows")
        return solve_test_rows();
    if (argc != 7) {
        cerr << "school_portfolio NUMERIC PRIVATE YEAR BATCH OUTPUT KERNEL\n";
        return 2;
    }
    const string kernel = argv[6];
    if (kernel != "independent" && kernel != "common")
        throw runtime_error("unknown simulation kernel");
    const int year = stoi(argv[3]), batch = stoi(argv[4]);
    string prefix = string(argv[1]) + "/" + argv[3], aux = string(argv[2]) + "/" + argv[3];
    Market m(prefix + ".bin");
    ifstream tierfile(aux + "-tiers", ios::binary);
    auto tier = readv<unsigned char>(tierfile, m.ne);
    if (tierfile.peek() != EOF)
        throw runtime_error("extra priority tier data");
    const auto recorded = m.priority;
    SimulatedPriority sampler(m, tier, year, batch);
    vector<char> all(m.ne, 1);
    vector<int> counts(m.ne);
    vector<Word> attainable(m.ne);
    for (int draw = 1; draw <= 64; ++draw) {
        if (kernel == "independent")
            sampler.draw(m.priority, draw);
        else
            common_draw(m.priority, m, tier, year, batch, draw);
        auto match = m.da(all);
        auto accept = cutoff_attainable(m, match);
        for (int e : match)
            if (e >= 0)
                ++counts[e];
        Word bit = Word(1) << (draw - 1);
        for (int e = 0; e < m.ne; ++e)
            if (accept[e])
                attainable[e] |= bit;
    }
    Menus candidates(m, counts, attainable);
    m.priority = recorded;
    auto full = m.da(all);
    auto released = cutoff_attainable(m, full);
    auto mc_match = m.da(candidates.mc);
    ifstream registry(aux + "-register", ios::binary);
    auto rbd = readv<int>(registry, m.m), april = readv<int>(registry, m.n);
    if (registry.peek() != EOF)
        throw runtime_error("extra register data");
    ofstream out(argv[5]);
    if (!out)
        throw runtime_error("cannot write aggregate results");
    out << "year,batch,policy,applicants,placed,historical_shown,full_shown,full_named,submitted_"
           "first,linked,linked_placed,april_match,no_official,no_official_placed,no_official_"
           "linked,no_official_april,gained_vs_mc,lost_vs_mc,no_official_gained_vs_mc,no_official_"
           "lost_vs_mc,changed_vs_mc,changed_vs_ties,unchanged_gained_vs_mc,unchanged_lost_vs_mc,"
           "recorded_cutoff_coverage,simulation_coverage,simulation_partner_hits,simulation_"
           "attainable_count,april_shown,no_official_april_shown\n";
    for (string pol : {"mc", "ties", "coverage", "blind", "rank"}) {
        vector<char> menu;
        if (pol == "mc")
            menu = candidates.mc;
        else if (pol == "ties")
            menu = candidates.ties;
        else if (pol == "coverage")
            menu = candidates.coverage;
        else {
            ifstream file(prefix + "-" + pol + ".mask", ios::binary);
            menu = readv<char>(file, m.ne);
        }
        evaluate(out, year, batch, pol, m, menu, candidates.mc, candidates.ties, mc_match, full,
                 released, counts, attainable, rbd, april);
    }
}
