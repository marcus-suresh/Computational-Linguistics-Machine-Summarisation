# Computational Linguistics - Machine Summarisation
This repository contains the Python source codes using NLP techniques such as Term Frequency–Inverse Document Frequency (TF-IDF) algorithm as well as other NLP text processing methods to generate machine summarisations of input text based on a compression score. The higher the compression, the more concise the output summary will be. 

Below are some outputs of machine summarisation using speech data.

# Treasury Speech x1.2 compression (Low Threshold - Verbose)
```
However, this will still be the single biggest economic shock Australia 
has faced in living memory. As the Secretary of the Treasury, I take 
full responsibility for the revised costing of the JobKeeper program and 
all matters associated with the advice that Treasury has provided. Costing approach for JobKeeper
The JobKeeper program was deliberately designed as a demand driven program. 
This means the extent of the program would flex in response to the need for the program. 
As is the case for any demand driven program, the level of uncertainty around the individual program’s 
costing is larger than other program designs. When Treasury costed the JobKeeper program two methods were used. 
We did assess that given the design of the program and the size of the subsidy, businesses had a strong incentive 
to participate in the program. On 3 April, National Cabinet continued to focus on the central assumption of 6 months 
of restrictions. However, the level of restrictions were broadly maintained, and there was not a move to tighten 
further to allow only essential activities. Such a move would have restricted activities such as construction, 
manufacturing and non-essential retail more broadly. As decisions evolved around the degree of economic restrictions, 
assessments of the likely severity of the economic contraction also evolved. This was in line with subsequent announcements 
by other forecasters. We also began to receive information about the economic impact, from data and through the business 
liaison program. The economic outlook will continue to evolve. Roll out of the JobKeeper program
The Jobkeeper program opened for enrolment on 20 April, following strong interest from business through the 
ATO register of interest. Enrolments increased steadily from that period onwards, as did the indications of 
the number of employees covered. Despite the economic outlook not being as severe as assumed in the costing, 
the reported JobKeeper enrolments data from the ATO tracked steadily towards the original costings estimate. 
First businesses had to enrol in the program after assessing they met the turnover test. At this point, they 
provided an estimate to the ATO of the number of eligible employees they were likely to have. The enrolment 
data which were the subject of the reporting error, were collected to provide an early estimate of the number 
of expected employees likely to access the JobKeeper program. That over 99 per cent of forms were correctly 
completed, suggests the form was well-designed. 2 - COVID-19 Daily Cases, sourced from the Department of Health.
```

# Treasury Speech x1.3 compression (Medium threshold)
```
However, this will still be the single biggest economic shock Australia has faced in living memory. 
Costing approach for JobKeeper
The JobKeeper program was deliberately designed as a demand driven program. As is the case for 
any demand driven program, the level of uncertainty around the individual program’s costing is 
larger than other program designs. We did assess that given the design of the program and the 
size of the subsidy, businesses had a strong incentive to participate in the program. However, 
the level of restrictions were broadly maintained, and there was not a move to tighten further 
to allow only essential activities. Such a move would have restricted activities such as construction, 
manufacturing and non-essential retail more broadly. This was in line with subsequent announcements
by other forecasters. We also began to receive information about the economic impact, from data and
through the business liaison program. The economic outlook will continue to evolve. Enrolments 
increased steadily from that period onwards, as did the indications of the number of employees covered.
At this point, they provided an estimate to the ATO of the number of eligible employees they were likely 
to have. That over 99 per cent of forms were correctly completed, suggests the form was well-designed. 
2 - COVID-19 Daily Cases, sourced from the Department of Health.
```

# Treasury Speech x1.4 compression (Medium-High Threshold)
```
However, this will still be the single biggest economic shock Australia has faced in living memory. 
Costing approach for JobKeeper
The JobKeeper program was deliberately designed as a demand driven program. 
We did assess that given the design of the program and the size of the subsidy, businesses had a 
strong incentive to participate in the program. However, the level of restrictions were broadly maintained, 
and there was not a move to tighten further to allow only essential activities. This was in line with 
subsequent announcements by other forecasters. We also began to receive information about the economic impact, 
from data and through the business liaison program. The economic outlook will continue to evolve. 
Enrolments increased steadily from that period onwards, as did the indications of the number of employees covered. 
At this point, they provided an estimate to the ATO of the number of eligible employees they were likely to have. 
That over 99 per cent of forms were correctly completed, suggests the form was well-designed.
```
# Treasury Speech x1.5 compression (High threshold)
```
However, this will still be the single biggest economic shock Australia has faced in living memory. 
However, the level of restrictions were broadly maintained, and there was not a move to tighten further 
to allow only essential activities. This was in line with subsequent announcements by other forecasters. 
The economic outlook will continue to evolve. That over 99 per cent of forms were correctly completed, 
suggests the form was well-designed.
```
