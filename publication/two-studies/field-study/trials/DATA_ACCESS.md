# Which Chilean field comparisons can be reproduced?

Agte, Allende, Kapor, Neilson and Ochoa report two randomized interventions in Chile’s 2021 admissions cycle. Published tables permit comparisons of application behavior, assignment and characteristics of the school attended. A test that the particular school shown to a family appeared in its application or became its enrolled school would require the family’s assigned arm and the identifiers of the schools it was shown. Our search found no public file linking those exposures to the administrative outcomes. [March 2026 revision](https://www.nber.org/system/files/working_papers/w32670/w32670.pdf) · [Working paper record](https://doi.org/10.3386/w32670)

## Public records checked on November 14, 2026

| Source | What it supplies |
| --- | --- |
| [AEA trial registration 7998](https://www.socialscienceregistry.org/trials/7998) | Describes the randomized Explorer intervention. Its data-publication field says “No” and its program-files field is empty. |
| [Working paper](https://www.nber.org/papers/w32670) and [coauthor’s publication listing](https://claudiaallende.com/research) | Paper and supplement. Neither page linked an archive of experimental microdata for this study when checked. |
| [Harvard Dataverse title query](https://dataverse.harvard.edu/api/search?q=%22Search%20and%20Biased%20Beliefs%20in%20Education%20Markets%22&type=dataset) and [DataCite title query](https://api.datacite.org/dois?query=titles.title%3A%22Search%20and%20Biased%20Beliefs%20in%20Education%20Markets%22) | Each returned zero datasets or records under that title. These searches cover only indexed deposits under matching titles. |
| [Ministry archive for applications in 2021 and admission in 2022](https://datosabiertos.mineduc.cl/wp-content/uploads/2022/02/SAE_2021.rar) | Eleven CSV tables. C1 has regular-round application rankings, and D1 has regular-round offers and responses. Across the eleven headers there is no trial arm, personalized report opening or set of displayed schools. D1’s offer and response fields cannot by themselves establish attendance. |

To reproduce the ministry check, supply the public archive and run the following command. The script verifies the input digest and writes the header names. No applicant records are written.

~~~sh
python3 inspect_2021_release.py --archive SAE_2021.rar --output sae-2021-public-schema.json
~~~

## Outcomes available in the paper

The early Explorer intervention compared a basic map, aggregate counts, and those counts plus highlighted schools and their prices and quality. A school met the highlighted category when its monthly price was below CLP 50,000 and its official quality rating was medium or high. Rank and enrollment tables use that category. They do not compare each family’s school IDs to its displayed recommendations. Different outcomes also have different samples. For example, 3,111 Explorer users enter the application-submission regression and 2,817 enter the nonassignment regression. [Table I and description](https://www.nber.org/system/files/working_papers/w32670/w32670.pdf#page=33) · [Appendix Table A.XVIII](https://www.nber.org/system/files/working_papers/w32670/w32670.pdf#page=86)

The late report listed omitted schools within two kilometers of home and sorted them by quality. The same report gave feedback on existing choices and sometimes included a risk warning. Its published enrollment outcomes are attributes of the school attended. The randomized invitation also instrumented the reaction to opening this bundle. The added-school outcome measures whether the family added any school, whose identity is unreported. [Table II](https://www.nber.org/system/files/working_papers/w32670/w32670.pdf#page=35)
