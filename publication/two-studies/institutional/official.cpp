#include <map>
#define main original_replay
#include "replay.cpp"
#undef main
#include "institution.hpp"
int main(int argc,char** argv){
    if(argc!=5){cerr<<"usage: official PREFIX SIDECAR YEAR OUTPUT\n";return 2;}
    string prefix=argv[1];Market m(prefix+".bin");Institutional ins(m,argv[2]);
    ofstream out(argv[4]);out<<"year,menu,group,n,official_offers,placed,correct_program_or_outside,correct_school_or_outside,correct_placement_status,correct_program_offered,official_only,replay_only,wrong_program_both,different_school_both,no_official_or_replay,blocking_display_pairs\n";
    for(string name:{"full","blind","rank"}){
        vector<char> menu(m.ne,1);if(name!="full"){ifstream in(prefix+"-"+name+".mask",ios::binary);menu=readv<char>(in,m.ne);}
        auto result=ins.clear(menu);map<string,array<int,13>> rows;
        for(int i=0;i<m.n;++i){
            int e=result.partner[i],p=e<0?-1:m.prog[e],q=m.official[i];
            bool both=p>=0&&q>=0,school=both&&ins.school[p]==ins.school[q],correct=p==q;
            int blocks=0;bool family=!ins.relatives[i].empty(),special=false;
            for(int f=m.at[i];f<m.at[i+1];++f){
                if(menu[f])blocks+=ins.blocks(i,f,result);
                int j=m.prog[f];special|=ins.capacity[4*j+2]>0||ins.capacity[4*j+3]>0;
            }
            if(blocks)throw runtime_error("displayed contract blocks detailed DA");
            array<int,13> one={1,q>=0,p>=0,correct,school||(p<0&&q<0),(p<0)==(q<0),both&&correct,q>=0&&p<0,p>=0&&q<0,both&&!correct,both&&!school,p<0&&q<0,blocks};
            for(const string& s:vector<string>{"all",family?"higher_grade_link":"no_higher_grade_link",m.ret[i]>=0?"continuity":"no_continuity",special?"PIE_or_academic_program_on_report":"no_PIE_or_academic_program_on_report",m.ses[i]?"disadvantaged":"other",string("grade_")+to_string(m.grade[i])})
                for(size_t h=0;h<one.size();++h)rows[s][h]+=one[h];
        }
        for(const auto& g:rows){out<<argv[3]<<','<<name<<','<<g.first;for(auto val:g.second)out<<','<<val;out<<'\n';}
    }
}
