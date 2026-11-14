// Score the harm of removing each old-only course. A held edge used without
// a viable worse alternative on a baseline draw is costly to drop. For adds,
// prefer a course with no historical baseline block. All scores are guidance
// for hindsight menu search only; final outputs must pass independent DA.
#define CEILING_LIBRARY
#include "institution_ceiling.cpp"
int main(int argc,char** argv){
    if(argc!=9)throw runtime_error("prefix year tiers register side out draws batch");
    string prefix=argv[1];int year=stoi(argv[2]);Market m(prefix+".bin");
    ifstream in(argv[3],ios::binary);auto tier=readv<unsigned char>(in,m.ne);
    Masks menus(m,prefix,argv[4]);Institutional r(m,argv[5]);r.set_simulated();
    vector<double> score(m.ne);SimulatedPriority rng(m,tier,year,stoi(argv[8]));Interleave inter(m,tier,year,stoi(argv[8]));
    for(int d=1;d<=stoi(argv[7]);++d){
      rng.draw(m.priority,d);inter.apply(m,d);
      auto base=r.clear(menus.a);
      for(const auto& row:menus.rows){
        if(row.baseline_hit[1])continue;int i=row.applicant,h=base.partner[i];
        bool positive=false;for(int e:row.proposed)positive|=menus.hit(1,i,e);if(!positive)continue;
        for(int e:row.proposed)if(menus.hit(1,i,e))score[e]+=r.blocks(i,e,base);
        if(h<0||menus.b[h])continue;
        bool safe=false;
        for(int e=m.at[i];e<m.at[i+1];++e){
            if(!menus.a[e]||e==h)continue;
            // The following is a ranking heuristic, so hypothetical family-priority
            // changes are permitted. Any eventual certification uses the full engine.
            int j=m.prog[e];
            if(r.flags[e]&Institutional::Current){safe=true;break;}
            for(int c:r.route(i,e,base)){
                int g=4*j+c;
                if(r.capacity[g] && (base.occupancy[g]<r.capacity[g] || r.score(i,e,c,base)<base.worst[g])) safe=true;
            }
        }
        score[h]+=safe?1:10;
      }
    }
    ofstream out(argv[6],ios::binary);out.write(reinterpret_cast<char*>(score.data()),score.size()*sizeof(double));
    if(!out)throw runtime_error("score write");
}
