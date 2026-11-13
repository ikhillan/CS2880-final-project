// Restrict the original five-slot policy to selected editors; never export identifiers to the results bundle.
#define main old_main
#include "replay.cpp"
#undef main
int main(int argc,char** argv){
    if(argc!=4)return 2;
    Market m(argv[1]);int year=stoi(argv[2]);string dir=argv[3],prefix=dir+"/"+to_string(year)+"-";
    mt19937_64 rng(28311+year*1000);
    vector<uint64_t> tie(m.ne);generate(tie.begin(),tie.end(),[&]{return rng();});
    for(string pol:{"full","blind","rank"}){
        auto mask=pol=="full"?vector<char>(m.ne,1):m.menu(pol=="blind"?m.blind:m.rank,tie,5,pol);
        auto ed=m.da(mask);m.verify(ed,mask);
        vector<int> prog(m.n);for(int i=0;i<m.n;i++)prog[i]=ed[i]<0?-1:m.prog[ed[i]];
        ofstream a(prefix+pol+".mask",ios::binary),b(prefix+pol+".partners",ios::binary);
        a.write(mask.data(),m.ne);b.write(reinterpret_cast<char*>(prog.data()),m.n*sizeof(int));
    }
    return 0;
}
