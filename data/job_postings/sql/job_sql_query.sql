SELECT
    revelio_job_postings.postings_cosmos.rcid,
    revelio_job_postings.postings_cosmos.post_date,
    revelio_job_postings.postings_cosmos.job_id,
    revelio_job_postings.postings_cosmos.company,
    revelio_job_postings.postings_cosmos.country,
    revelio_job_postings.postings_cosmos.role_k1500_v2,
    revelio_job_postings.postings_cosmos.state,
    revelio_job_postings.postings_cosmos.role_k17000_v3,
    revelio_job_postings.postings_cosmos.metro_area,
    revelio_job_postings.postings_cosmos.onet_code,
    revelio_job_postings.postings_cosmos.seniority,
    revelio_job_postings.postings_cosmos.salary,
    revelio_job_postings.postings_cosmos.salary_min,
    revelio_job_postings.postings_cosmos.salary_max,
    revelio_job_postings.postings_cosmos.salary_predicted,
    revelio_job_postings.postings_cosmos.remove_date,
    revelio_job_postings.postings_cosmos.remote_type,
    revelio_job_postings.postings_cosmos.expected_hires

FROM revelio_job_postings.postings_cosmos

WHERE revelio_job_postings.postings_cosmos.post_date BETWEEN
    '2025-01-01'::date AND '2025-12-31'::date

AND ("revelio_job_postings"."postings_cosmos"."country" = 'United States' AND "revelio_job_postings"."postings_cosmos"."salary_predicted" = false AND "revelio_job_postings"."postings_cosmos"."onet_code" IS NOT NULL AND "revelio_job_postings"."postings_cosmos"."metro_area" = 'new york city metropolitan area')
