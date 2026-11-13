#define main verified_menu_main
#include "menus.cpp"
#undef main
#include <random>

int main(int argc,char** argv) {
    if(argc!=3) return 1;
    std::ifstream in(argv[1],std::ios::binary); std::ofstream out(argv[2]);
    std::int32_t number;read(in,number);
    out<<"market,weight,lower_n,higher_n,switched,baseline,perm_sum_delta\n";
    for(int i=0;i<number;++i) {
        Market m;read(in,m);
        int nl=std::count(m.low.begin(),m.low.end(),1);
        std::vector<int> high;
        for(int b=0;b<n;++b) if(!m.low[b]) high.push_back(b);
        auto base=shortlist(m.uniform,m.pref);
        const std::array<Order,2> targets={shortlist(m.score1,m.pref),shortlist(m.score4,m.pref)};
        int before=lower_count(solve(base,m.priority),m.low);
        std::array<std::vector<int>,2> totals;
        for(auto& v:totals)v.assign(high.size()+1,0);
        for(int rep=0;rep<24;++rep) {
            std::seed_seq seed{2607861,i,rep};
            std::mt19937_64 rng(seed);
            auto sequence=high;
            // Explicit rejection sampling makes the permutation portable across c++ libraries.
            for(size_t k=sequence.size();k>1;--k) {
                uint64_t threshold=(uint64_t(0)-k)%k,draw;
                do{draw=rng();}while(draw<threshold);
                std::swap(sequence[k-1],sequence[draw%k]);
            }
            for(int w=0;w<2;++w) {
                auto graph=base;
                for(size_t k=0;k<sequence.size();++k) {
                    const int b=sequence[k]; graph[b]=targets[w][b];
                    totals[w][k+1]+=lower_count(solve(graph,m.priority),m.low)-before;
                }
            }
        }
        for(int w=0;w<2;++w) for(size_t k=0;k<totals[w].size();++k) {
            out<<i<<','<<(w?4:1)<<','<<nl<<','<<high.size()<<','<<k<<','<<before<<','<<totals[w][k]<<'\n';
        }
    }
    return 0;
}
