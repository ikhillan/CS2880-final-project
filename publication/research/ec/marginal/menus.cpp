// Displacement menus on the calibrated 64-by-64 markets. Input is packed by run.py.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int n = 64, k = 5;
using Row = std::array<std::uint8_t,n>;
using Matrix = std::array<Row,n>;
using Flags = std::array<bool,n>;
using Order = std::array<std::vector<int>,n>;
using Vec = std::array<int,n>;

struct State {
    Vec held, next{};
    State() { held.fill(-1); }
};

template<class T> void read(std::ifstream& in, T& a) {
    in.read(reinterpret_cast<char*>(&a),sizeof(a));
    if (!in) throw std::runtime_error("Truncated rank input");
}
void check(bool b) { if (!b) throw std::runtime_error("DA verification failed"); }

void propose(State& state, const Order& lists, const Matrix& priority, int applicant) {
    while (applicant >= 0 && state.next[applicant] < int(lists[applicant].size())) {
        int seat = lists[applicant][state.next[applicant]++];
        int incumbent = state.held[seat];
        if (incumbent < 0 || priority[applicant][seat] < priority[incumbent][seat]) {
            state.held[seat] = applicant;
            applicant = incumbent;
        }
    }
}

State solve(const Order& lists,const Matrix& priority,int omitted=-1) {
    State s;
    for (int a=0; a<n; ++a) if (a!=omitted) propose(s,lists,priority,a);
    return s;
}
Vec assignments(const State& s) {
    Vec out; out.fill(-1);
    for (int j=0;j<n;++j) if (s.held[j]>=0) out[s.held[j]]=j;
    return out;
}
int lower_count(const State& s,const Row& low) {
    int z=0;
    for(int a:s.held) if(a>=0) z+=low[a];
    return z;
}
int single_label(const State& absent,const Order& lists,const Matrix& priority,int h,int j) {
    // -2: unattainable, -1: no displacement, 0..63: incumbent losing a seat.
    State s=absent;
    int loser=s.held[j];
    if(loser>=0 && priority[h][j]>priority[loser][j]) return -2;
    s.held[j]=h;
    while(loser>=0 && loser!=h && s.next[loser]<int(lists[loser].size())) {
        int dest=lists[loser][s.next[loser]++];
        int other=s.held[dest];
        if(other<0 || priority[loser][dest]<priority[other][dest]) {
            s.held[dest]=loser; loser=other;
        }
    }
    return loser==h ? -2 : loser;
}

int chosen(const std::vector<int>& sorted,const Vec& labels) {
    for (int j:sorted) if (labels[j]!=-2) return j;
    return -1;
}
double choose(int a,int b) {
    if(a<b || b<0) return 0;
    double res=1;
    for(int i=1;i<=b;++i) res=res*(a-i+1)/i;
    return res;
}

struct Market {
    Matrix pref,priority,uniform,score1,score4;
    Row low;
};

Order shortlist(const Matrix& exposure,const Matrix& pref) {
    Order out;
    for(int a=0;a<n;++a) {
        for(int j=0;j<n;++j) if(exposure[a][j]<k) out[a].push_back(j);
        std::sort(out[a].begin(),out[a].end(),[&](int x,int y){return pref[a][x]<pref[a][y];});
    }
    return out;
}

void analyze(int market,const Market& m,std::ostream& out) {
    auto base=shortlist(m.uniform,m.pref);
    std::array<Order,2> policy={shortlist(m.score1,m.pref),shortlist(m.score4,m.pref)};
    State initial=solve(base,m.priority);
    Vec orig=assignments(initial);
    int nl=std::count(m.low.begin(),m.low.end(),1),before=lower_count(initial,m.low);
    std::array<int,2> loss{},gain{};
    std::array<double,2> expected{};
    int probes=0, direct_checks=0;
    for(int h=0;h<n;++h) {
        if(m.low[h]) continue;
        State absent=solve(base,m.priority,h);
        Vec central=assignments(absent),labels; labels.fill(-2);
        std::vector<int> attainable;
        int degree=0;
        for(int j=0;j<n;++j) if(m.pref[h][j]<n) {
            ++degree;++probes;
            labels[j]=single_label(absent,base,m.priority,h,j);
            if(labels[j]!=-2) attainable.push_back(j);
            if(market<24) {
                Order singleton=base; singleton[h]={j};
                auto direct=assignments(solve(singleton,m.priority));
                check((direct[h]==j)==(labels[j]!=-2));
                for(int b=0;b<n;++b) if(b!=h) {
                    check((direct[b]>=0)==(central[b]>=0 && b!=labels[j]));
                }
                ++direct_checks;
            }
        }
        std::sort(attainable.begin(),attainable.end(),[&](int x,int y){return m.pref[h][x]<m.pref[h][y];});
        double u_expected=0,mass=0;
        int count=std::min(degree,k);
        for(size_t r=0;r<attainable.size();++r) {
            double pr=choose(degree-int(r)-1,count-1)/choose(degree,count);
            int j=attainable[r]; mass+=pr;
            if(labels[j]>=0 && m.low[labels[j]]) u_expected+=pr;
        }
        check(std::abs(mass+choose(degree-int(attainable.size()),count)/choose(degree,count)-1.)<1e-10);
        int u=chosen(base[h],labels);
        check(u==orig[h]);
        for(int w=0;w<2;++w) {
            int t=chosen(policy[w][h],labels);
            int ul=(u>=0 && labels[u]>=0)?labels[u]:-1;
            int tl=(t>=0 && labels[t]>=0)?labels[t]:-1;
            if(ul!=tl) {
                if(ul>=0 && m.low[ul]) ++gain[w];
                if(tl>=0 && m.low[tl]) ++loss[w];
            }
            expected[w]+=u_expected-double(tl>=0 && m.low[tl]);
            if(market<24) {
                auto lists=base; lists[h]=policy[w][h];
                auto direct=assignments(solve(lists,m.priority));
                check(direct[h]==t);
                for(int b=0;b<n;++b) if(b!=h) check((direct[b]>=0)==(central[b]>=0 && b!=tl));
                ++direct_checks;
            }
        }
    }
    for(int w=0;w<2;++w) {
        Order joint=base;
        for(int a=0;a<n;++a) if(!m.low[a]) joint[a]=policy[w][a];
        auto final=solve(joint,m.priority);auto end=assignments(final);
        int after=lower_count(final,m.low),entered=0,exited=0,lost=0,gained=0;
        Order residual;
        for(int j=0;j<n;++j) {
            bool init_h=initial.held[j]>=0 && !m.low[initial.held[j]];
            bool end_h=final.held[j]>=0 && !m.low[final.held[j]];
            entered+=end_h&&!init_h;exited+=init_h&&!end_h;
        }
        for(int a=0;a<n;++a) if(m.low[a]) {
            lost+=orig[a]>=0 && end[a]<0; gained+=orig[a]<0 && end[a]>=0;
            for(int j:base[a]) if(final.held[j]<0 || m.low[final.held[j]]) residual[a].push_back(j);
        }
        auto predicted=assignments(solve(residual,m.priority));
        for(int a=0;a<n;++a) if(m.low[a])check((end[a]>=0)==(predicted[a]>=0));
        check(lost<=entered && gained<=exited);
        int path_losses[2]{},path_gains[2]{};
        for(int reverse=0;reverse<2;++reverse) {
            Order path=base;
            auto prev=orig;
            for(int i=0;i<n;++i) {
                int a=reverse?n-1-i:i;
                if(m.low[a])continue;
                path[a]=policy[w][a];
                auto next=assignments(solve(path,m.priority));
                int pl=0,pg=0;
                for(int b=0;b<n;++b) if(m.low[b]) {pl+=prev[b]>=0 && next[b]<0;pg+=prev[b]<0 && next[b]>=0;}
                check(pl<=1 && pg<=1);
                path_losses[reverse]+=pl;path_gains[reverse]+=pg;prev=next;
            }
            check(prev==end && path_gains[reverse]-path_losses[reverse]==after-before);
        }
        out<<market<<','<<(w==0?1:4)<<','<<nl<<','<<before<<','<<after<<','<<probes<<','<<direct_checks<<','
           <<loss[w]<<','<<gain[w]<<','<<expected[w]<<','<<lost<<','<<gained<<','<<entered<<','<<exited<<','
           <<path_losses[0]<<','<<path_gains[0]<<','<<path_losses[1]<<','<<path_gains[1]<<'\n';
    }
}

int main(int argc,char** argv) {
    if(argc!=3)return 1;
    std::ifstream in(argv[1],std::ios::binary);std::ofstream out(argv[2]);int32_t count;read(in,count);
    check(out.good());out<<std::setprecision(15);
    out<<"market,weight,lower_n,initial,final,probes,direct_checks,isolated_loss,isolated_gain,expected_net,"
          "joint_loss,joint_gain,newly_high_seats,released_high_seats,forward_loss,forward_gain,reverse_loss,reverse_gain\n";
    for(int i=0;i<count;++i) {Market m;read(in,m);analyze(i,m,out);}
    check(in.peek()==EOF);
    return 0;
}
