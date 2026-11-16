// Serial, renter-at-a-time deferred acceptance for an independent count replay.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

constexpr int N = 64;
using Rank = std::uint8_t;

template <typename T>
void read_array(std::ifstream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated tensor file");
}

void write_count(std::ofstream& output, int count) {
    const std::array<std::uint8_t,2> bytes{static_cast<std::uint8_t>(count), 0};
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

int main(int argc,char** argv) {
    if (argc != 4) return 2;
    const bool crossed = std::string(argv[3]) == "crossed";
    std::ifstream input(argv[1], std::ios::binary);
    std::ofstream output(argv[2], std::ios::binary);
    std::int32_t market_count;
    read_array(input,market_count);
    for (int market=0; market<market_count; ++market) {
        std::array<Rank,9*N*N> buyer;
        std::array<Rank,N*N> seller, uniform;
        std::array<Rank,3*N*N> transparent;
        std::array<Rank,N> lower;
        read_array(input,buyer);
        read_array(input,seller);
        read_array(input,uniform);
        read_array(input,transparent);
        read_array(input,lower);
        for (int grid=0; grid<9; ++grid) {
            const Rank* ranks = buyer.data() + N*N*(crossed ? 3*(grid/3) : grid);
            const Rank* score = transparent.data() + N*N*(crossed ? grid%3 : grid/3);
            for (int budget=0; budget<2; ++budget) {
                std::array<int,14> counts{};
                bool assigned[3][N]{};
                bool fully_visible[N]{};
                for (int arm=0; arm<3; ++arm) {
                    int options[N][N], partner[N], holder[N], free_renters[N];
                    int next[N]{}, length[N]{}, free_count=0;
                    for (int b=0; b<N; ++b) {
                        int degree=0;
                        for (int s=0; s<N; ++s) degree += ranks[b*N+s]<N;
                        const int quota = budget==0 ? std::min(5,degree) : (2*degree+4)/5;
                        fully_visible[b] = lower[b] && degree>0 && degree==quota;
                        std::fill(options[b], options[b]+N, -1);
                        for (int s=0; s<N; ++s) {
                            const bool use_uniform = arm==0 || (arm==1 && lower[b]);
                            const Rank visible_rank = use_uniform ? uniform[b*N+s] : score[b*N+s];
                            if (visible_rank<quota) options[b][ranks[b*N+s]]=s;
                        }
                        for (int rank=0; rank<N; ++rank) {
                            if (options[b][rank]>=0) options[b][length[b]++]=options[b][rank];
                        }
                        holder[b]=partner[b]=-1;
                        free_renters[free_count++]=b;
                    }
                    while (free_count>0) {
                        const int b=free_renters[--free_count];
                        while (next[b]<length[b]) {
                            const int s=options[b][next[b]++];
                            const int incumbent=holder[s];
                            if (incumbent<0 || seller[b*N+s]<seller[incumbent*N+s]) {
                                if (incumbent>=0) {
                                    partner[incumbent]=-1;
                                    free_renters[free_count++]=incumbent;
                                }
                                partner[b]=s;
                                holder[s]=b;
                                break;
                            }
                        }
                    }
                    for (int b=0; b<N; ++b) {
                        bool blocked=false;
                        const int cutoff=partner[b]<0 ? 126 : ranks[b*N+partner[b]];
                        for (int s=0; s<N; ++s) {
                            if (ranks[b*N+s]<cutoff &&
                                (holder[s]<0 || seller[b*N+s]<seller[holder[s]*N+s])) blocked=true;
                        }
                        counts[arm*4+int(!lower[b])] += partner[b]>=0;
                        counts[arm*4+2+int(!lower[b])] += blocked;
                        assigned[arm][b] = partner[b]>=0;
                    }
                }
                for (int b=0; b<N; ++b) {
                    if (!fully_visible[b]) continue;
                    counts[12] += assigned[0][b] && !assigned[1][b];
                    counts[13] += !assigned[0][b] && assigned[1][b];
                }
                for (int count:counts) write_count(output,count);
            }
        }
    }
    return !output.good();
}
