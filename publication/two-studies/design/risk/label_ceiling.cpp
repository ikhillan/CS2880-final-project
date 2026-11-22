// Separate hindsight ceilings on newly covered historical offers and April schools.
#define LOTTERY_DESIGN_LIBRARY
#include "lottery_design.cpp"

int main(int argc,char** argv) {
    if(argc!=6) {cerr<<"usage: ceiling NUMERIC YEAR TIERS REGISTER OUTPUT\n";return 2;}
    int year=stoi(argv[2]);string prefix=string(argv[1])+"/"+argv[2];
    Market m(prefix+".bin");
    ifstream input(argv[3],ios::binary),reg(argv[4],ios::binary);
    auto tier=readv<unsigned char>(input,m.ne);
    auto schools=readv<int>(reg,m.m),april=readv<int>(reg,m.n);
    if(reg.peek()!=EOF || input.peek()!=EOF)throw runtime_error("input length mismatch");
    Design design(m,prefix);SimulatedPriority train(m,tier,year,513);
    ofstream out(argv[5]);if(!out)throw runtime_error("output failure");
    out<<"year,support,target,present_at_baseline,additional_reachable\n";
    for(int d=0;d<=1024;++d){
        if(d){train.draw(m.priority,d);design.learn(Baseline(m,design.a),1);}
        if(d!=0 && d!=16 && d!=64 && d!=256 && d!=1024)continue;
        for(bool school:{false,true}){
            int shown=0,additional=0;
            for(int i=0;i<m.n;++i){
                int label=school?april[i]:m.official[i];
                if(label<0)continue;
                bool baseline_hit=false, candidate=false, can_remove=false;
                for(int e=m.at[i];e<m.at[i+1];++e){
                    bool hit=(school?schools[m.prog[e]]:m.prog[e])==label;
                    baseline_hit|=design.a[e]&&hit;
                    can_remove|=design.a[e]&&!design.b[e]&&!design.required[e];
                    candidate|=!design.a[e]&&design.b[e]&&!design.forbidden[e]&&hit;
                }
                shown+=baseline_hit;additional+=!baseline_hit&&candidate&&can_remove;
            }
            out<<year<<','<<d<<','<<(school?"april_school":"initial_offer")<<','<<shown<<','<<additional<<'\n';
        }
    }
}
