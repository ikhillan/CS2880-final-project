# Does an accurate offer prediction cover a program the student later attends?

University entrant and enrollment records are official PAES releases for 2025 and 2026 from the MINEDUC open-data education portal. Use the RAR archives listed in sources.json. Keep individual inputs and joins local. The shared outputs contain code, neural weights, aggregate counts and compute receipts; there are no applicant identities or individual enrollment outcomes. PROTOCOL.md was written before inspecting linkage results. TRAINING-FOLLOWUP.md documents the later change of training objective on the same test cohort. The first analysis uses networks previously fit on actual 2025 offers. The second changes the observed 2025 training target to the enrolled program among the applicant's valid regular options and keeps the information, training hyperparameters, seeds and held-out year fixed. Outside here means no linked registration on that list. Enrollment is observed under the historical policy. A counterfactual placement is the result of rerunning DA after pruning reported choices.

Rebuild the previous university arrays, frozen predictions and index mapping using research/observed/university. Run fetch_enrollment.py ENROLL_ROOT to download and verify the enrollment RARs. Substitute ENROLL_ROOT/2025 and ENROLL_ROOT/2026 as appropriate. PROTOCOL.md explains which registrants enter each denominator. The published files each have one distinct enrolled program per linked person; the code asserts uniqueness in the 2026 file and checks duplicates for 2025. Aggregate receipt files include full input SHA256 checksums. Use the same OR-Tools environment as the adjacent frontier for the shared menu helper.

~~~sh
P=research/observed/university
F=research/policy/frontier
E=research/policy/enrollment
python "$E/check.py" "$P" "$F" INPUTS ENROLL_2026 FEATURES OFFER_FIT replay/offer-label
python "$E/retarget.py" "$P" INPUTS ENROLL_2025 FEATURES replay/enrollment_features
cp FEATURES/features2026.npz replay/enrollment_features/
python "$P/replay_offer.py" replay/enrollment_features "$E/models" replay/enroll-fit --device cuda
python "$P/ensemble.py" replay/enroll-fit
python "$E/compare.py" "$P" "$F" INPUTS ENROLL_2026 FEATURES replay/enroll-fit replay/enrollment-label replay/enrollment_features
~~~

To refit the enrollment models use the six fixed task indices in the previous university train_offer.py and its Slurm array, pointing --data to replay/enrollment_features and --out to a fresh fit directory. No 2026 outcome is supplied to the forward pass. Hardware and library settings can affect numerical near-ties when regenerating scores.
