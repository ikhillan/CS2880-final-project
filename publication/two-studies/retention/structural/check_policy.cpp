// Reclear precomputed structural menus and the blind baseline with identical priorities.
// Input labels are used only for historical coverage after every mask has been fixed.
#define CEILING_LIBRARY
#include "institution_ceiling.cpp"

vector<char> read_menu(const string& path, int length) {
    ifstream in(path,ios::binary);
    auto v=readv<char>(in,length);
    if(in.peek()!=EOF)throw runtime_error("unexpected menu size");
    return v;
}

int main(int argc, char** argv){
    if(argc!=10){
        cerr<<"check_policy PREFIX YEAR TIERS REG SIDE PROSPECTIVE ORACLE OUTPUT DRAWS\n";
        return 2;
    }
    string prefix=argv[1];int year=stoi(argv[2]),draws=stoi(argv[9]);
    Market m(prefix+".bin"); Masks masks(m,prefix,argv[4]); Institutional inst(m,argv[5]);
    ifstream input(argv[3],ios::binary);auto tier=readv<unsigned char>(input,m.ne);
    if(input.peek()!=EOF)throw runtime_error("tier size");
    const vector<vector<char>> menus={read_menu(argv[6],m.ne),read_menu(argv[7],m.ne)};
    vector<int> delta;
    for(const auto& row:menus) {
        int gain=0;
        for(int i=0;i<m.n;++i){
            bool hit=false;
            int quota=0;
            for(int e=m.at[i];e<m.at[i+1];++e){
                if(row[e] && masks.hit(1,i,e))hit=true;
                if((masks.a[e] && masks.b[e] && !row[e]) || (!masks.a[e] && !masks.b[e] && row[e]))throw runtime_error("candidate constraint");
                quota+=row[e]-masks.a[e];
            }
            if(quota)throw runtime_error("quota"); gain+=hit;
        }
        delta.push_back(gain-masks.shown[1]);
    }
    SimulatedPriority gen(m,tier,year,90267319);
    Interleave inter(m,tier,year,90267319);
    ofstream out(argv[8]);
    out<<"year,split,draw,rule,baseline_placed,exits,entries,course_switches,school_switches,april_hit_difference\n";
    for(int d=0;d<=draws;++d){
        if(d){
            gen.draw(m.priority,d);inter.apply(m,d);inst.set_simulated();
        }else inst.set_recorded();
        auto old=inst.clear(masks.a);
        for(int k=0;k<int(menus.size());++k){
            auto now=inst.clear(menus[k]);
            int seats=0,exits=0,entries=0,course=0,school=0;
            for(int i=0;i<m.n;++i){
                int a=old.partner[i],b=now.partner[i];
                seats+=a>=0;exits+=a>=0&&b<0;entries+=a<0&&b>=0;
                if(a>=0&&b>=0){course+=a!=b;school+=inst.school[m.prog[a]]!=inst.school[m.prog[b]];}
            }
            out<<year<<','<<(d?"fresh":"recorded")<<','<<d<<','<<(k?"oracle":"prospective")<<','<<seats<<','<<exits<<','<<entries<<','<<course<<','<<school<<','<<delta[k]<<'\n';
            if(exits)throw runtime_error("structural protection failed");
        }
    }
    if(!out)throw runtime_error("output write");
}
