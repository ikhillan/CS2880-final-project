// Private checksum audit against independent simultaneous DA. No person-level output is distributed.
#define main primary_replay_main
#include "replay.cpp"
#undef main
int main(int argc,char** argv){
    if(argc!=4)return 2;
    Market a(argv[1]);int year=stoi(argv[2]);string dest(argv[3]);
    mt19937_64 rng(28311+year*1000);vector<uint64_t> ties(a.ne);generate(ties.begin(),ties.end(),[&]{return rng();});
    for(string pol:{"full","first","blind","rank"}){
        auto mask=pol=="full"?vector<char>(a.ne,1):a.menu(pol=="rank"?a.rank:a.blind,ties,5,pol);
        auto edge=a.da(mask);a.verify(edge,mask);vector<int> out(a.n);
        for(int i=0;i<a.n;i++)out[i]=edge[i]<0?-1:a.prog[edge[i]];
        ofstream masks(dest+"/"+to_string(year)+"-"+pol+".mask",ios::binary);
        masks.write(mask.data(),a.ne);
        ofstream values(dest+"/"+to_string(year)+"-"+pol+".partners",ios::binary);
        values.write(reinterpret_cast<const char*>(out.data()),a.n*sizeof(int));
    }
    return 0;
}
