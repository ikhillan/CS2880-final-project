// Include after the public replay.cpp Market definition. Input has no official outcomes.
#pragma once
#include <array>
#include <limits>
#include <tuple>
#include <utility>

struct Institutional {
    enum Category { Regular=0, Disadvantaged=1, Pie=2, Academic=3, Fallback=4 };
    enum Flag { Current=1, Sibling=2, Staff=4, Former=8, PriorPie=16 };
    struct Score {
        int tier, special, lottery, applicant;
        bool operator<(const Score& rhs) const {
            return tie(tier,special,lottery,applicant)<tie(rhs.tier,rhs.special,rhs.lottery,rhs.applicant);
        }
    };
    struct Outcome {
        vector<int> partner,category,occupancy;
        vector<Score> worst;
        vector<unsigned char> recent_sibling, promoted;
    };
    struct Next { int cursor=0,pass=0,category=0; };
    struct Link { int older; bool sibling,block; };
    const Market& m;
    vector<int> school,grade,capacity,lottery,pie_order,academic_order,owner;
    vector<unsigned char> honors,flags,transitional;
    vector<vector<Link>> relatives;
    array<vector<int>,14> grades;
    bool simulation=false;
    Institutional(const Market& market,const string& path):m(market){
        ifstream in(path,ios::binary); auto h=readv<int>(in,4);
        if(h[0]!=m.n || h[1]!=m.m || h[2]!=m.ne)throw runtime_error("institution sidecar differs from market");
        school=readv<int>(in,m.m);grade=readv<int>(in,m.m);
        auto raw=readv<int>(in,5*m.m);
        capacity.resize(m.m*4);transitional.resize(m.m);
        for(int j=0;j<m.m;++j){
            for(int k=0;k<3;++k) capacity[4*j+k]=raw[5*j+k];
            capacity[4*j+Academic]=raw[5*j+3]+raw[5*j+4];
            transitional[j]=raw[5*j+3]>0;
        }
        honors=readv<unsigned char>(in,m.n);flags=readv<unsigned char>(in,m.ne);
        lottery=readv<int>(in,m.ne);pie_order=readv<int>(in,m.ne);academic_order=readv<int>(in,m.ne);
        auto links=readv<int>(in,4*h[3]);relatives.resize(m.n);
        for(int l=0;l<h[3];++l)relatives.at(links[l*4+1]).push_back({links[l*4],bool(links[l*4+2]),bool(links[l*4+3])});
        if(in.peek()!=EOF)throw runtime_error("extra sidecar contents");
        owner.resize(m.ne);
        for(int i=0;i<m.n;++i){
            grades.at(m.grade[i]+1).push_back(i);
            fill(owner.begin()+m.at[i],owner.begin()+m.at[i+1],i);
        }
    }
    void set_recorded(){simulation=false;}
    void set_simulated(){simulation=true;}
    Score score(int i,int e,int c,const Outcome& o) const {
        int j=m.prog[e];
        int tier=7,rank=0;
        bool eligible=c==Pie?pie_order[e]>=0:c==Academic?(transitional[j]?academic_order[e]>=0:honors[i]):false;
        if(flags[e]&Current)tier=0;
        else if(eligible){
            tier=1;
            if(c==Pie)rank=pie_order[e];
            if(c==Academic && transitional[j])rank=academic_order[e];
        }else if(flags[e]&Sibling)tier=2;
        else if(o.recent_sibling[e])tier=3;
        else if(c==Disadvantaged && m.ses[i])tier=4;
        else if(flags[e]&Staff)tier=5;
        else if(flags[e]&Former)tier=6;
        return {tier,rank,simulation?m.priority[e]:lottery[e],i};
    }
    const array<int,4>& route(int i,int e,const Outcome& o) const {
        static constexpr array<int,4> non_dis_sib={Pie,Academic,Regular,Disadvantaged};
        static constexpr array<int,4> non_others={Pie,Academic,Disadvantaged,Regular};
        static constexpr array<int,4> ret_pie_dis={Pie,Disadvantaged,Regular,Academic};
        static constexpr array<int,4> ret_pie={Pie,Regular,Disadvantaged,Academic};
        static constexpr array<int,4> ret_dis={Disadvantaged,Regular,Academic,Pie};
        static constexpr array<int,4> ret_other={Regular,Disadvantaged,Academic,Pie};
        if(flags[e]&Current){
            if(flags[e]&PriorPie)return m.ses[i]?ret_pie_dis:ret_pie;
            return m.ses[i]?ret_dis:ret_other;
        }
        return !m.ses[i]&&((flags[e]&Sibling)||o.recent_sibling[e])?non_dis_sib:non_others;
    }
    // A grade's higher-grade destinations remain constant throughout that grade's DA.
    Outcome clear(const vector<char>& menu) const {
        Outcome o;
        o.partner.assign(m.n,-1);o.category.assign(m.n,-1);
        o.occupancy.assign(capacity.size(),0);o.worst.resize(capacity.size());
        o.recent_sibling.assign(m.ne,0);o.promoted.assign(m.ne,0);
        vector<Next> n(m.n);
        vector<unsigned char> any_promoted(m.n);
        struct Held {Score priority;int i;};
        auto better=[](const Held& a,const Held& b){return a.priority<b.priority;};
        vector<vector<Held>> pools(capacity.size());
        for(auto g=grades.rbegin();g!=grades.rend();++g){
            for(int i:*g){
                for(Link link:relatives[i]){
                    int p=o.partner[link.older];if(p<0)continue;
                    int family_school=school[m.prog[p]];
                    for(int e=m.at[i];e<m.at[i+1];++e)if(school[m.prog[e]]==family_school){
                        o.recent_sibling[e]|=link.sibling;
                        o.promoted[e]|=link.block;
                        any_promoted[i]|=link.block&&menu[e];
                    }
                }
                n[i].cursor=m.at[i]; n[i].pass=any_promoted[i]?0:1;
            }
            for(int first:*g){
                int i=first;
                while(i>=0){
                    auto& next=n[i];
                    if(next.cursor>=m.at[i+1]){
                        if(next.pass==0){++next.pass;next.cursor=m.at[i];next.category=0;continue;}
                        break;
                    }
                    int e=next.cursor,j=m.prog[e];
                    if(!menu[e] || (next.pass==0)!=bool(o.promoted[e])){
                        ++next.cursor;next.category=0;continue;
                    }
                    if(next.category==4){
                        if(flags[e]&Current){o.partner[i]=e;o.category[i]=Fallback;break;}
                        ++next.cursor;next.category=0;continue;
                    }
                    int cat=route(i,e,o)[next.category++],sub=4*j+cat,cap=capacity[sub];
                    if(!cap)continue;
                    Held candidate{score(i,e,cat,o),i};
                    auto& pool=pools[sub];int evicted=-1;
                    if(int(pool.size())==cap){
                        if(!better(candidate,pool.front()))continue;
                        pop_heap(pool.begin(),pool.end(),better);
                        evicted=pool.back().i;pool.pop_back();
                        o.partner[evicted]=-1;o.category[evicted]=-1;
                    }
                    pool.push_back(candidate);push_heap(pool.begin(),pool.end(),better);
                    o.partner[i]=e;o.category[i]=cat;i=evicted;
                }
            }
        }
        for(size_t s=0;s<capacity.size();++s){
            o.occupancy[s]=int(pools[s].size());
            if(!pools[s].empty())o.worst[s]=pools[s].front().priority;
        }
        return o;
    }
    // Test a whole applicant-course pair against the detailed baseline. Same-program
    // categories are ordered; a holder who would rather have another type can block there.
    bool blocks(int i,int e,const Outcome& o) const {
        int held=o.partner[i];bool same=e==held;
        if(!same&&held>=0){
            if(o.promoted[e]!=o.promoted[held]){if(!o.promoted[e])return false;}
            else if(e>held)return false;
        }
        int j=m.prog[e];
        for(int cat:route(i,e,o)){
            if(same&&cat==o.category[i])return false;
            int sub=4*j+cat;
            if(capacity[sub] && (o.occupancy[sub]<capacity[sub] || score(i,e,cat,o)<o.worst[sub]))return true;
        }
        return !same && (flags[e]&Current);
    }
};
