// All allowed edges of a capacitated bipartite matching covering the incumbent set.
#pragma once
struct HallFilter {
    vector<int> component, base_node;
    vector<unsigned char> reaches_spare;
    vector<vector<pair<int,int>>> forced;
    vector<unsigned char> forced_course_sensitive;
    vector<unsigned char> after_target;
    int courses;
    HallFilter(const LowerCutoffs &o,const Institutional::Outcome &baseline,const Institutional::Outcome &ghost)
       :base_node(o.m.n,-1),courses(o.m.m) {
        const auto &m=o.m; const auto &rule=o.rule;
        const int nodes=courses+1, private_pool=courses;
        vector<vector<int>> to(nodes),from(nodes);
        vector<int> cap(nodes),used(nodes);
        for(int j=0;j<courses;++j)for(int c=0;c<4;++c)cap[j]+=rule.capacity[4*j+c];
        // All pupils with a protected personal fallback get their own unlimited
        // outside-of-physical-capacity place, accessible only to such pupils.
        cap[private_pool]=m.n+1;
        for(int i=0;i<m.n;++i) {
            int e=baseline.partner[i];if(e<0)continue;
            base_node[i]=(baseline.category[i]==Institutional::Fallback)?private_pool:m.prog[e];
            ++used[base_node[i]];
        }
        for(int i=0;i<m.n;++i) {
            int a=base_node[i];if(a<0)continue;
            for(int e=m.at[i];e<m.at[i+1];++e) {
                if(!(o.menus.a[e]||o.menus.b[e]) || o.rejected(i,e,ghost))continue;
                int j=m.prog[e];
                if(j!=a)to[a].push_back(j);
                if((rule.flags[e]&Institutional::Current) && a!=private_pool)
                    to[a].push_back(private_pool);
            }
        }
        for(int j=0;j<nodes;++j) {
            sort(to[j].begin(),to[j].end());
            to[j].erase(unique(to[j].begin(),to[j].end()),to[j].end());
            for(int k:to[j])from[k].push_back(j);
        }
        vector<unsigned char> visited(nodes);
        vector<int> order,stack;
        for(int j=0;j<nodes;++j) {
            if(visited[j])continue;
            stack.push_back(j);
            while(!stack.empty()){
                int u=stack.back();stack.pop_back();
                if(u<0){order.push_back(~u);continue;}
                if(visited[u])continue;
                visited[u]=1;stack.push_back(~u);
                for(int v:to[u])if(!visited[v])stack.push_back(v);
            }
        }
        if(int(order.size())!=nodes)throw runtime_error("SCC visit counting");
        component.assign(nodes,-1);
        int id=0;
        for(auto it=order.rbegin();it!=order.rend();++it) {
            int root=*it;if(component[root]>=0)continue;
            stack.push_back(root);component[root]=id;
            while(!stack.empty()){
                int u=stack.back();stack.pop_back();
                for(int v:from[u])if(component[v]<0){component[v]=id;stack.push_back(v);}
            }
            ++id;
        }
        reaches_spare.assign(nodes,0);
        for(int j=0;j<nodes;++j) {
            if(used[j]>cap[j])throw runtime_error("Baseline exceeds relaxed capacity");
            if(used[j]<cap[j]){stack.push_back(j);reaches_spare[j]=1;}
        }
        while(!stack.empty()){
            int u=stack.back();stack.pop_back();
            for(int v:from[u])if(!reaches_spare[v]){reaches_spare[v]=1;stack.push_back(v);}
        }
        forced.resize(courses);
        forced_course_sensitive.assign(courses,0);
        after_target.assign(m.ne,0);
        vector<int> cap_physical(courses);
        for(int j=0;j<courses;++j)cap_physical[j]=cap[j];
        for(int i=0;i<m.n;++i) {
            int a=base_node[i];
            if(a<0 || a==private_pool)continue;
            int edge=-1;
            bool other=false;
            for(int e=m.at[i];e<m.at[i+1];++e) {
                if(!(o.menus.a[e]||o.menus.b[e]) || o.rejected(i,e,ghost) || !permits(o,i,e))continue;
                if(rule.flags[e]&Institutional::Current){other=true;break;}
                if(m.prog[e]!=a){other=true;break;}
                edge=e;
            }
            if(other || edge<0)continue;
            forced[a].push_back({i,edge});
        }
        for(int j=0;j<courses;++j) {
            if(int(forced[j].size())>cap_physical[j])throw runtime_error("Too many forced physical pupils");
            forced_course_sensitive[j]=cap_physical[j]>0 && int(forced[j].size())==cap_physical[j];
        }
        Institutional::Outcome ordinary;
        ordinary.recent_sibling.assign(m.ne,0);
        for(const auto &d:o.designs){
            for(int e=m.at[d.pupil];e<m.at[d.pupil+1];++e){
                if(!(o.menus.a[e]||o.menus.b[e]))continue;
                int j=m.prog[e];
                if(!forced_course_sensitive[j])continue;
                const auto &occupants=forced[j];
                if(find_if(occupants.begin(),occupants.end(),[&](const auto &v){return v.first==d.pupil;})!=occupants.end())
                    continue;
                // Someone among the forced incumbents loses their assigned seat to i
                // at every possible category they might occupy, whenever i proposes.
                bool veto=false;
                for(auto [h,he]:occupants){
                    bool above=true;
                    for(int c=0;c<4;++c)if(rule.capacity[4*j+c] &&
                        !(rule.score(d.pupil,e,c,ordinary)<rule.score(h,he,c,o.favorable))){above=false;break;}
                    if(above){veto=true;break;}
                }
                after_target[e]=veto;
            }
        }
    }
    bool permits(const LowerCutoffs &o,int i,int e)const {
        if(o.rule.flags[e]&Institutional::Current)return true;
        int j=o.m.prog[e],a=base_node[i];
        return reaches_spare[j] || (a>=0 && component[j]==component[a]);
    }
};
