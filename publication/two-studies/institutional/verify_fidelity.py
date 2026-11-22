"""Cross-check annual D1 denominators, strata, transitions and reserve-rule ablations."""
import argparse
import csv
from pathlib import Path

def read(path):
    with path.open() as stream:
        return [{k:int(v) if v.lstrip("-").isdigit() else v for k,v in row.items()} for row in csv.DictReader(stream)]

def main(args):
    here=args.engine
    baseline_mistakes=[];official_offers=[]; stage_deltas=[]
    all_pupils=[];linked=[];without=[]
    fit=[0]*5
    for year in range(2022,2026):
        tiers=read(here/f"ablation-{year}.csv")
        transition={r["stratum"]:r for r in read(here/f"transitions-{year}.csv")}
        exact={(r["menu"],r["group"]):r for r in read(here/f"official-{year}.csv")}
        assert len(tiers)==5
        pooled=tiers[0]
        for i,r in enumerate(tiers):
            assert r["correct_program_or_outside"]==pooled["correct_program_or_outside"]+r["corrected_vs_pooled"]-r["new_errors_vs_pooled"]
            fit[i]+=r["correct_program_or_outside"]
        for r in exact.values():
            assert r["correct_program_or_outside"]==r["n"]-r["official_only"]-r["replay_only"]-r["wrong_program_both"]
            assert r["correct_school_or_outside"]==r["n"]-r["official_only"]-r["replay_only"]-r["different_school_both"]
            assert r["blocking_display_pairs"]==0
        full=exact['full','all']
        assert full['correct_program_or_outside']==tiers[-1]['correct_program_or_outside']
        for r in transition.values():
            assert r["rank_placed"]-r["blind_placed"]==r["gross_entries"]-r["gross_exits"]
        for mainrow,partition in (("all",("initial_offer","no_initial_offer")),("all",("changed_menu","identical_menu")),
                                  ("all",("higher_link","no_higher_link")),
                                  ("all_linked",("linked_no_offer","enrolled_at_initial_offer","enrolled_elsewhere"))):
            for key,total in transition[mainrow].items():
                if key not in ("year","stratum"):
                    assert total==sum(transition[group][key] for group in partition)
        observed={name:{r['group']:r for r in read(args.official/f"{name}-{year}.csv")} for name in ('all','cutoff','linked')}
        for strata in (observed['all'],observed['cutoff']):
            for row in strata.values():
                assert row['same_course_or_outside']+row['official_only']+row['replay_only']+row['both_placed_different_course']==row['applicants']
        for row in observed['all'].values():
            assert row['same_course_or_outside']==row['applicants']==row['same_offer_status']
        assert observed['all']['all']['applicants']==full['n']
        official_offers.append(observed['all']['official_offer']['applicants'])
        assert observed['cutoff']['all']['same_course_or_outside']==full['correct_program_or_outside']
        baseline_mistakes.append(full['n']-full['correct_program_or_outside'])
        stage_deltas.append([tiers[k]['correct_program_or_outside']-pooled['correct_program_or_outside'] for k in range(5)])
        for local,reference in (('all_linked','linked'),('linked_no_offer','no_initial_offer'),('enrolled_at_initial_offer','enrolled_at_initial_offer'),('enrolled_elsewhere','enrolled_elsewhere')):
            left=transition[local];right=observed['linked'][reference]
            for a,b in [('n','linked'),('gross_exits','exits'),('gross_entries','entrants'),('blind_placed','blind_placed'),('rank_placed','rank_placed'),('blind_april_shown','blind_hits'),('rank_april_shown','rank_hits')]:
                assert left[a]==right[b],(year,local,a,left[a],right[b])
        for design in ("blind","rank"):assert transition['all'][design+'_placed']==exact[design,'all']['placed']
        all_pupils.append(transition["all"]); linked.append(transition["all_linked"]); without.append(transition["linked_no_offer"])
    def summary(rows):
        return tuple(sum(r[k] for r in rows) for k in ("n","gross_exits","gross_entries"))+(sum(r["rank_april_shown"]-r["blind_april_shown"] for r in rows),)
    assert official_offers==[520420,492888,439038,430712]
    assert baseline_mistakes==[16,39,42,44]
    assert all(d[0]==0 and d[3]<=d[4] for d in stage_deltas)
    assert summary(all_pupils)==(2044649,2635,801,4638)
    assert summary(linked)==(1969120,2397,727,4638)
    assert summary(without)==(145450,1549,279,-43)
    print("Four cohorts, D1 error decompositions, mechanism ladder, policy losses and enrollment strata verified.")

if __name__=="__main__":
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine",type=Path,default=root/"results")
    parser.add_argument("--official",type=Path,default=root/"results")
    main(parser.parse_args())
