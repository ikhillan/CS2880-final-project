// Construct and independently test a prospective, seat-retaining menu.
// Uses the archived institutional engine and its exact matched-course lotteries.
#define CEILING_LIBRARY
#include "institution_ceiling.cpp"
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <iomanip>

struct Inputs {
    Market m;
    Institutional inst;
    vector<char> a,b;
    vector<int> switched, candidates, potential, seats;
    Inputs(const string& prefix,const string& side):m(prefix+".bin"),inst(m,side) {
        // The shared serialized Market includes offline targets. Explicitly mask
        // them and the recorded lottery before any fitting, screening or audit.
        fill(m.official.begin(),m.official.end(),-1);
        fill(m.priority.begin(),m.priority.end(),0);
        fill(inst.lottery.begin(),inst.lottery.end(),0);
        inst.set_simulated();
        a=read_mask(prefix+"-blind.mask",m.ne);
        b=read_mask(prefix+"-rank.mask",m.ne);
        potential.resize(m.m);seats.resize(m.m);
        for(int j=0;j<m.m;++j) seats[j]=accumulate(inst.capacity.begin()+4*j,inst.capacity.begin()+4*j+4,0);
        for(int i=0;i<m.n;++i) {
            int na=0,nb=0;bool changed=false;
            for(int e=m.at[i];e<m.at[i+1];++e) {
                if(a[e]||b[e])++potential[m.prog[e]];
                if(a[e]!=b[e]){changed=true;candidates.push_back(e);}
                na+=a[e];nb+=b[e];
                if(m.prog[e]==m.ret[i]&&!(a[e]&&b[e]))throw runtime_error("omitted continuity");
            }
            if(na!=nb)throw runtime_error("inconsistent quota");
            if(changed)switched.push_back(i);
        }
    }
    static vector<char> read_mask(const string& path,int n) {
        ifstream in(path,ios::binary);
        auto v=readv<char>(in,n);
        if(in.peek()!=EOF)throw runtime_error("mask length mismatch");
        return v;
    }
};

struct Draws {
    Inputs& inputs;
    vector<unsigned char> tier;
    unique_ptr<SimulatedPriority> courses;
    unique_ptr<Interleave> interleaving;
    unique_ptr<FamilySchoolLottery> families;
    Draws(Inputs& x,const string& tiers,int year,int batch,const string& family):inputs(x) {
        ifstream in(tiers,ios::binary);tier=readv<unsigned char>(in,x.m.ne);
        if(in.peek()!=EOF)throw runtime_error("tier file length");
        if(family=="-"){
            courses=make_unique<SimulatedPriority>(x.m,tier,year,batch);
            interleaving=make_unique<Interleave>(x.m,tier,year,batch);
        }else families=make_unique<FamilySchoolLottery>(x.m,family,year,batch);
    }
    void apply(int draw) {
        if(families)families->draw(inputs.m.priority,draw);
        else {courses->draw(inputs.m.priority,draw);interleaving->apply(inputs.m,draw);}
    }
};

struct Feature {
    int best=1000000;
    double blind=-1e100,rank=-1e100;
};

int probability_bin(double logit){
    static const array<double,4> boundaries={log(.1/.9),log(.25/.75),0.,log(.75/.25)};
    return upper_bound(boundaries.begin(),boundaries.end(),logit)-boundaries.begin();
}

int feature_cell(const Feature& f){
    int r=min(f.best-1,6);
    if(f.best==7||f.best==8)r=5;
    return (r*5+probability_bin(f.blind))*5+probability_bin(f.rank);
}

map<int,int> school_features(const Inputs& x,int i) {
    map<int,Feature> values;
    for(int e=x.m.at[i];e<x.m.at[i+1];++e)if(x.a[e]||x.b[e]){
        auto& f=values[x.inst.school[x.m.prog[e]]];
        f.best=min(f.best,e-x.m.at[i]+1);f.blind=max(f.blind,x.m.blind[e]);f.rank=max(f.rank,x.m.rank[e]);
    }
    map<int,int> result;
    for(auto [school,f]:values)result[school]=feature_cell(f);
    return result;
}

vector<int> read_april(const Inputs& x,const string& reg){
    ifstream in(reg,ios::binary);
    if(readv<int>(in,x.m.m)!=x.inst.school)throw runtime_error("register school mismatch");
    auto v=readv<int>(in,x.m.n);
    if(in.peek()!=EOF)throw runtime_error("register length mismatch");
    return v;
}

void train_calibration(const Inputs& x,const string& reg,const string& out){
    auto april=read_april(x,reg);
    array<int,175> n{},hits{};
    for(int i:x.switched)if(april[i]>=0){
        for(auto [school,cell]:school_features(x,i)){++n[cell];hits[cell]+=school==april[i];}
    }
    ofstream f(out);f<<"cell,rank_bin,blind_bin,informed_bin,school_pairs,enrolled,probability\n";
    int total=accumulate(n.begin(),n.end(),0),win=accumulate(hits.begin(),hits.end(),0);
    for(int r=0;r<7;++r){
        int rn=accumulate(n.begin()+r*25,n.begin()+(r+1)*25,0);
        int rw=accumulate(hits.begin()+r*25,hits.begin()+(r+1)*25,0);
        double prior=(rw+40.*win/total)/(rn+40.);
        for(int k=25*r;k<25*r+25;++k)
            f<<k<<','<<r<<','<<k%25/5<<','<<k%5<<','<<n[k]<<','<<hits[k]<<','<<setprecision(17)<<(hits[k]+40.*prior)/(n[k]+40.)<<'\n';
    }
    if(!f)throw runtime_error("write failed");
}

void observe(Inputs& x,const Institutional::Outcome& old,vector<char>& excluded) {
    for(int i:x.switched)for(int e=x.m.at[i];e<x.m.at[i+1];++e){
        if(x.a[e]==x.b[e]||excluded[e])continue;
        int p=old.partner[i];
        if(x.m.grade[i]==-1){
            if(x.a[e])excluded[e]=p==e;
            else if(x.potential[x.m.prog[e]]>x.seats[x.m.prog[e]])
                excluded[e]=x.inst.blocks(i,e,old);
        }else {
            if(p<0)excluded[e]=true;
            else if(old.promoted[e]!=old.promoted[p])excluded[e]=bool(old.promoted[e]);
            else excluded[e]=e<=p;
        }
    }
}

void screen(Inputs& x,const string& tiers,int year,int first,int last,const string& out){
    Draws generator(x,tiers,year,513,"-");
    vector<char> excluded(x.m.ne);
    for(int d=first;d<=last;++d){
        generator.apply(d);observe(x,x.inst.clear(x.a),excluded);
    }
    ofstream f(out,ios::binary);f.write(excluded.data(),excluded.size());
    if(!f)throw runtime_error("write screen");
}

array<double,175> read_model(const string& path){
    ifstream in(path);string row;getline(in,row);
    array<double,175> result{};
    for(int k=0;k<175;++k){
        if(!getline(in,row))throw runtime_error("short calibration file");
        replace(row.begin(),row.end(),',',' ');
        istringstream input(row);int cell,rank,blind,informed,n,h;
        if(!(input>>cell>>rank>>blind>>informed>>n>>h>>result[k])||cell!=k)throw runtime_error("invalid calibration");
    }
    return result;
}

void choose_menu(const Inputs& x,const vector<char>& excluded,const string& model_path,const string& out){
    auto probs=read_model(model_path);
    vector<char> menu=x.a;
    for(int i:x.switched){
        auto features=school_features(x,i);
        set<int> represented;
        vector<int> available,old_edges,new_edges;
        int quota=0;
        for(int e=x.m.at[i];e<x.m.at[i+1];++e){
            if(x.a[e]){++quota;old_edges.push_back(e);}
            if((x.a[e]&&x.b[e])||(x.a[e]&&excluded[e])){new_edges.push_back(e);represented.insert(x.inst.school[x.m.prog[e]]);}
            else if(x.a[e]||(x.b[e]&&!excluded[e]))available.push_back(e);
        }
        auto key=[&](int e){
            int s=x.inst.school[x.m.prog[e]];
            return make_tuple(-probs[features.at(s)],!x.a[e],e);
        };
        sort(available.begin(),available.end(),[&](int e,int f){return key(e)<key(f);});
        for(int e:available){
            if(int(new_edges.size())>=quota)break;
            int s=x.inst.school[x.m.prog[e]];
            if(represented.insert(s).second)new_edges.push_back(e);
        }
        if(int(new_edges.size())<quota){
            sort(available.begin(),available.end(),[&](int e,int f){return make_pair(!x.a[e],e)<make_pair(!x.a[f],f);});
            for(int e:available){
                if(int(new_edges.size())==quota)break;
                if(find(new_edges.begin(),new_edges.end(),e)==new_edges.end())new_edges.push_back(e);
            }
        }
        if(int(new_edges.size())!=quota)throw runtime_error("unfillable quota");
        auto value=[&](const vector<int>& edges){
            set<int> schools;
            for(int e:edges)schools.insert(x.inst.school[x.m.prog[e]]);
            long double sum=0;
            for(int s:schools)sum+=probs[features.at(s)];
            return sum;
        };
        if(value(new_edges)>value(old_edges)+1e-14L){
            fill(menu.begin()+x.m.at[i],menu.begin()+x.m.at[i+1],0);
            for(int e:new_edges)menu[e]=1;
        }
    }
    ofstream f(out,ios::binary);f.write(menu.data(),menu.size());
    if(!f)throw runtime_error("write menu");
}

void audit(Inputs& x,const string& tiers,int year,int batch,const string& family,int first,int last,
           const string& frozen,const string& out,bool with_rank){
    auto policy=Inputs::read_mask(frozen,x.m.ne);
    Draws generator(x,tiers,year,batch,family);
    ofstream f(out);
    f<<"year,architecture,batch,draw,policy,baseline_placed,exits,entries,course_switch,school_switch,category_only,retained_better,retained_worse,placed\n";
    for(int d=first;d<=last;++d){
        generator.apply(d);
        auto old=x.inst.clear(x.a);
        for(int arm=0;arm<=(with_rank?1:0);++arm){
            auto newer=x.inst.clear(arm==0?policy:x.b);
            int baseline=0,exits=0,entries=0,course=0,school=0,category=0,better=0,worse=0;
            for(int i=0;i<x.m.n;++i){
                int p=old.partner[i],q=newer.partner[i];
                baseline+=(p>=0);exits+=(p>=0&&q<0);entries+=(p<0&&q>=0);
                if(p<0||q<0)continue;
                course+=p!=q;school+=x.inst.school[x.m.prog[p]]!=x.inst.school[x.m.prog[q]];
                category+=p==q&&old.category[i]!=newer.category[i];
                better+=q<p;worse+=q>p;
            }
            f<<year<<','<<(family=="-"?"matched_course":"school_family")<<','<<batch<<','<<d<<','<<(arm==0?"learned_retention":"unrestricted_rank")<<','
             <<baseline<<','<<exits<<','<<entries<<','<<course<<','<<school<<','<<category<<','<<better<<','<<worse<<','<<baseline-exits+entries<<'\n';
            if(batch==513&&arm==0&&exits)throw runtime_error("design incumbent lost a seat");
        }
    }
    if(!f)throw runtime_error("write audit");
}

void evaluate(const Inputs& x,const string& frozen,const string& reg,int year,const string& path){
    auto april=read_april(x,reg);auto menu=Inputs::read_mask(frozen,x.m.ne);
    ofstream f(path);
    f<<"year,group,pupils,linked,baseline_hits,selected_hits,rank_hits,new_hits,lost_hits,pupils_menu_changed,substitutions\n";
    for(int g=0;g<3;++g){
        int n=0,linked=0,old=0,pol=0,b=0,gain=0,loss=0,changed=0,subs=0;
        for(int i=0;i<x.m.n;++i){
            if(g==1&&x.m.grade[i]!=-1)continue;
            if(g==2&&x.m.grade[i]==-1)continue;
            int hits[3]={0,0,0},adds=0;
            ++n;linked+=april[i]>=0;
            for(int e=x.m.at[i];e<x.m.at[i+1];++e){
                if((x.a[e]&&x.b[e]&&!menu[e])||(!x.a[e]&&!x.b[e]&&menu[e]))throw runtime_error("ineligible menu");
                if(april[i]>=0&&x.inst.school[x.m.prog[e]]==april[i]){
                    hits[0]|=x.a[e];hits[1]|=menu[e];hits[2]|=x.b[e];
                }
                adds+=menu[e]&&!x.a[e];
            }
            old+=hits[0];pol+=hits[1];b+=hits[2];gain+=hits[1]&&!hits[0];loss+=hits[0]&&!hits[1];
            changed+=adds>0;subs+=adds;
        }
        f<<year<<','<<vector<string>{"all","pre_k","higher"}[g]<<','<<n<<','<<linked<<','<<old<<','<<pol<<','<<b<<','<<gain<<','<<loss<<','<<changed<<','<<subs<<'\n';
    }
    if(!f)throw runtime_error("write outcomes");
}

#ifndef RETENTION_LIBRARY
int main(int argc,char** argv){
    if(argc<5)throw runtime_error("see README for arguments");
    string mode=argv[1];Inputs x(argv[2],argv[3]);
    if(mode=="fit"&&argc==6)train_calibration(x,argv[4],argv[5]);
    else if(mode=="screen"&&argc==9)screen(x,argv[4],stoi(argv[5]),stoi(argv[6]),stoi(argv[7]),argv[8]);
    else if(mode=="choose"&&argc==7)choose_menu(x,Inputs::read_mask(argv[4],x.m.ne),argv[5],argv[6]);
    else if(mode=="audit"&&argc==13)audit(x,argv[4],stoi(argv[5]),stoi(argv[6]),argv[7],stoi(argv[8]),stoi(argv[9]),argv[10],argv[11],stoi(argv[12])!=0);
    else if(mode=="score"&&argc==8)evaluate(x,argv[4],argv[5],stoi(argv[6]),argv[7]);
    else throw runtime_error("invalid arguments");
}
#endif
