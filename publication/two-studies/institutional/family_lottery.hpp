#pragma once
// Market and readv are defined by the common school replay.
#include <array>
#include <numeric>
#include <random>
#include <tuple>
class FamilySchoolLottery {
    int year;
    unsigned batch;
    struct Group { std::vector<int> pupils; };
    struct Pupil { std::vector<int> edges; };
    std::vector<std::vector<int>> school_groups;
    std::vector<Group> groups;
    std::vector<Pupil> applicants;
    template<class RNG> static void shuffle_values(std::vector<int>& ids, RNG& rng) {
        for(size_t n=ids.size(); n>1; --n) {
            uint64_t limit=(uint64_t(0)-uint64_t(n))%uint64_t(n),x;
            do x=rng();while(x<limit);
            std::swap(ids[n-1],ids[x%n]);
        }
    }
public:
    FamilySchoolLottery(const Market& m, const std::string& path, int yr,unsigned b):year(yr),batch(b){
        std::ifstream input(path,std::ios::binary);
        auto sizes=readv<int>(input,2);
        if(sizes[0]!=m.n||sizes[1]!=m.m)throw std::runtime_error("family lottery dimensions");
        auto family=readv<int>(input,m.n),schools=readv<int>(input,m.m);
        if(input.peek()!=EOF)throw std::runtime_error("family lottery input length");
        std::vector<std::array<int,4>> records;records.reserve(m.ne);
        for(int i=0;i<m.n;++i)for(int e=m.at[i];e<m.at[i+1];++e)
            records.push_back({schools[m.prog[e]],family[i],i,e});
        std::sort(records.begin(),records.end());
        std::array<int,4> last={-1,-1,-1,-1};
        for(const auto& a:records){
            bool new_school=a[0]!=last[0];
            bool new_family=new_school||a[1]!=last[1];
            if(new_school)school_groups.emplace_back();
            if(new_family){
                school_groups.back().push_back(static_cast<int>(groups.size()));
                groups.emplace_back();
            }
            if(new_family||a[2]!=last[2]){
                groups.back().pupils.push_back(static_cast<int>(applicants.size()));
                applicants.emplace_back();
            }
            applicants.back().edges.push_back(a[3]);
            last=a;
        }
    }
    void draw(std::vector<int>& tickets,int repetition) const{
        std::seed_seq seed{287461u,batch,static_cast<unsigned>(year),static_cast<unsigned>(repetition)};
        std::mt19937_64 rng(seed);
        std::vector<int> order,within;
        for(const auto& families:school_groups){
            order.assign(families.begin(),families.end());shuffle_values(order,rng);
            int rank=0;
            for(int g:order){
                const auto& members=groups[g].pupils;
                within.assign(members.begin(),members.end());shuffle_values(within,rng);
                for(int i:within){
                    for(int e:applicants[i].edges)tickets[e]=rank;
                    ++rank;
                }
            }
        }
    }
};
