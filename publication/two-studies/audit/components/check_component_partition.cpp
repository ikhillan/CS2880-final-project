// Independent disjoint-set calculation using an applicant/course graph.
// Usage: check_component_partition NATIVE_NUMERIC_DIRECTORY YEAR
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Id = int32_t;

template <class T>
std::vector<T> read(std::istream& file, size_t size) {
    std::vector<T> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    if (!file) throw std::runtime_error("incomplete input");
    return data;
}

class Sets {
    std::vector<Id> parent;
    std::vector<Id> weight;
public:
    explicit Sets(size_t size) : parent(size), weight(size, 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    Id root(Id a) {
        while (a != parent[a]) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    }
    void join(Id a, Id b) {
        a = root(a);
        b = root(b);
        if (a == b) return;
        if (weight[a] < weight[b]) std::swap(a,b);
        parent[b] = a;
        weight[a] += weight[b];
    }
};

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    const std::string dir = std::string(argv[1]) + "/" + argv[2];
    std::ifstream market(dir + ".bin", std::ios::binary);
    auto header = read<Id>(market, 3);
    const Id n = header[0], m = header[1], e = header[2];
    auto offsets = read<Id>(market, n + 1);
    auto courses = read<Id>(market, e);
    market.seekg(sizeof(Id) * (e + m), std::ios::cur);
    auto official = read<Id>(market, n);
    Sets sets(n+m);
    for (Id i=0; i<n; ++i) {
        if (official[i] >= 0) sets.join(i, n + official[i]);
        for (Id k=offsets[i]; k<offsets[i+1]; ++k) sets.join(i,n+courses[k]);
    }
    auto file = [&](const std::string& arm) {
        return std::ifstream(dir + "-" + arm, std::ios::binary);
    };
    auto full_file=file("full.partners"), blind_file=file("blind.partners"), informed_file=file("rank.partners");
    auto blind_mask=file("blind.mask"), informed_mask=file("rank.mask");
    auto full=read<Id>(full_file,n), blind=read<Id>(blind_file,n), informed=read<Id>(informed_file,n);
    auto initial=read<uint8_t>(blind_mask,e), revision=read<uint8_t>(informed_mask,e);
    std::vector<bool> disputed(n+m), visited(n+m);
    for (Id i=0;i<n;++i)
        if (official[i]!=full[i]) disputed[sets.root(i)]=true;
    int exact=0, components=0, changed=0, oldhits=0, newhits=0, entrants=0, exits=0;
    for (Id i=0;i<n;++i) {
        const Id r=sets.root(i);
        if (disputed[r]) continue;
        ++exact;
        if (!visited[r]) { visited[r]=true; ++components; }
        bool edited=false, a=false, b=false;
        for (Id k=offsets[i];k<offsets[i+1];++k) {
            edited |= initial[k]!=revision[k];
            if (courses[k]==official[i]) { a |= initial[k]; b |= revision[k]; }
        }
        changed+=edited;
        oldhits+=a && !b;
        newhits+=b && !a;
        entrants+=blind[i]<0 && informed[i]>=0;
        exits+=blind[i]>=0 && informed[i]<0;
    }
    std::cout << argv[2] << ',' << exact << ',' << components << ',' << changed << ',' << newhits << ',' << oldhits << ',' << entrants << ',' << exits << '\n';
}
