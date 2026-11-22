"""Check a stable old certificate and a newly displaced pupil under grade sequencing."""

def da(menus, order):
    assignment={}
    cursor={i:0 for i in menus}
    free=list(menus)
    while free:
        i=free.pop(0)
        if cursor[i]==len(menus[i]):
            continue
        j=menus[i][cursor[i]]
        cursor[i]+=1
        if j not in assignment:
            assignment[j]=i
        elif order[j].index(i)<order[j].index(assignment[j]):
            free.append(assignment[j])
            assignment[j]=i
        else:
            free.append(i)
    return {i:j for j,i in assignment.items()}

def stable(menus, order, assignment):
    holder={j:i for i,j in assignment.items()}
    for i,prefs in menus.items():
        old=assignment.get(i)
        if old is not None and old not in prefs: return False
        for j in prefs:
            if j==old: break
            if j not in holder or order[j].index(i)<order[j].index(holder[j]): return False
    return True

def main():
    # H,K,G are older. Each program has one seat per grade. H's underlying
    # report is x > u > y. In the new two-slot menu H gives up x for y.
    old={'H':['x','u'],'K':['x','y'],'G':['y','x']}
    new={'H':['u','y'],'K':['x','y'],'G':['y','x']}
    high={'x':['G','H','K'],'y':['K','G','H'],'u':['H']}
    baseline=da(old,high);updated=da(new,high)
    assert baseline=={'H':'u','K':'y','G':'x'}
    assert updated=={'H':'u','K':'x','G':'y'}
    # The old singleton list of junior L contains only x. Junior B ranks x then y.
    # With older K at y, B's sibling priority at y has no effect. With K at x, B
    # displaces L from x; the family's record has no linked-list promotion.
    junior={'L':['x'],'B':['x','y']}
    priorities_old={'x':['L','B'],'y':['B','L']}
    priorities_new={'x':['B','L'],'y':['B','L']}
    baseline_j=da(junior,priorities_old);updated_j=da(junior,priorities_new)
    assert stable(junior,priorities_old,baseline_j)
    assert baseline_j=={'L':'x','B':'y'} and updated_j=={'B':'x'}
    assert stable(new,high,baseline)
    print('A same-quota, no-block edit preserves the entire detailed assignment as a conditional witness but grade-sequential DA loses one of five prior recipients.')
if __name__=='__main__':
    main()
