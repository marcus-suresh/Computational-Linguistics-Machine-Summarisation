import nltk
nltk.download()
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

text_str = '''
Globally there have been over 7 million cases of COVID-19 and more than 400,000 deaths. The virus continues to spread in many countries delivering devastating health and economic outcomes.

Australia has had remarkably good health outcomes compared to many other countries. As the CMO noted to this committee, if Australia had experienced the same spread and fatality rates experienced in the UK, for example, instead of the 102 fatalities we have sadly seen, we might have seen significantly more fatalities.

Given the improved health outlook for Australia, the impact of COVID-19 on the economy will be smaller. However, this will still be the single biggest economic shock Australia has faced in living memory.

The JobKeeper program was designed and delivered to respond flexibly to the wide variety of health and economic scenarios that Australia faces as a result of COVID-19.

JobKeeper will be the largest fiscal stimulus program in Australia’s history. From the information we have to date, the program has been effective in mitigating job losses, and has kept millions of Australians attached to their employer in very difficult economic circumstances.

At the time the policy design and costing analysis for the JobKeeper program was being undertaken, the impacts of COVID-19 on the Australian population and economic activity were highly uncertain. When the program was announced on 30 March, the virus was spreading rapidly, both internationally and domestically, and health modelling based on Australia-specific observed transmission rates was not yet available.1 The full extent of measures needed to contain the spread of the virus domestically was not clear.

In this uncertain context, it was prudent to design the policy to be robust to whatever circumstances unfolded, to be demand driven, but to cost the JobKeeper policy under the assumption that very significant constraint measures would be required more akin to a lockdown. This economic scenario suggested GDP could fall by as much as 25 per cent in the June quarter. It is now clear that the fall in GDP for Australia is likely to be much less than this worst case scenario. It is a good outcome that unemployment is lower, and fewer businesses than originally expected are relying on Government support to pay their employees. JobKeeper was initially costed at $130 billion and we now expect it to cost around $70 billion.

As the Secretary of the Treasury, I take full responsibility for the revised costing of the JobKeeper program and all matters associated with the advice that Treasury has provided.

The remainder of this statement sets out the context for Treasury’s analysis at the time the JobKeeper policy was being developed and the subsequent events as they unfolded.

Health considerations
At the time the JobKeeper program was being finalised and costed, the virus caseload was increasing at the most rapid rate experienced to date in Australia.

It was spreading in multiple states and there had been a step up in untraceable community transmission.

The largest daily reported increase in cases (469) was reported on 28 March and the 5 day lagging average case load peaked on 28 March.2
For the three days prior to 30 March the case load compound growth rate was 12.3 per cent.
The Government was receiving epidemiological modelling on the possible health impact of COVID-19 in Australia, to inform transmission-reducing measures and health system preparedness.  Modelling by the Doherty Institute released publically by the Prime Minister on 7 April suggested that even when the virus was mitigated by targeted public health measures (quarantine and isolation), daily ICU demand was estimated to exceed 500 beds per million population in the median scenario.3
Social distancing policy considerations
Against the backdrop of rapidly increased transmission of the virus within Australia, the National Cabinet, guided by expert medical advice provided by the Australian Health Protection Principal Committee (AHPPC), imposed widespread social distancing restrictions across Australia on 24 March.

These restrictions included limiting the number of people allowed to congregate, with a maximum of 10 people at funerals and 5 people at weddings.
People were instructed to work from home where possible and not to travel unless for essential services.
These restrictions were a noticeable tightening from earlier restrictions imposed on 16, 18 and 20 March.
On 29 March the National Cabinet tightened gathering restrictions further to allow only up to 2 non‑family members to congregate in public spaces and the closure of public spaces such as playgrounds. People were instructed to stay in their homes unless doing essential shopping, going to work (if unable to work from home), seeking medical attention or undertaking personal outdoor exercise.

There was active consideration and widespread public discussion of a further tightening of restrictions to only allow a narrowly defined list of essential industries to be open, similar to the lock downs imposed in Italy, Spain, the United Kingdom, France and New Zealand.

Assessment of economic impact
In the context of the medical advice and the social distancing restrictions agreed by National Cabinet and announced on 24 March, Treasury provided briefing to the Treasurer on two economic scenarios on 27 March.

A scenario that assumed restrictions broadly consistent with those agreed by National Cabinet would be in place for 6 months.
This scenario suggested that GDP could be around 10 to 12 per cent below the MYEFO baseline in the June and September quarters, with around 2.1 million fewer people working4 across the economy over the six month period.
A scenario that assumed tighter restrictions where only a narrow set of essential industries and services would be allowed and such restrictions would be in place for 8 weeks before reverting to a lower level of restrictions for the remaining 4 months.
This scenario suggested GDP could be around 24 per cent below the MYEFO baseline in the June quarter, with around 4.8 million fewer people working5 across the economy over the 8 week period of the tighter restrictions.
This economic scenario was consistent with the advice provided by the OECD on the economic implications of the lockdown regimes in other nations. In those nations, where an essential industry focus was imposed, the economic loss was judged to be in the order of 20-30 per cent for the period of the lockdown.
Given the importance of understanding the potential funding needs of a program of the potential size and scope of the JobKeeper program, the trajectory that the virus and economic impacts were on at the time, and the sequential tightening of restrictions that had been imposed over the previous 2 weeks in Australia and overseas, it was judged prudent to cost JobKeeper using Treasury’s worse case economic scenario being modelled at the time.

I would note that falls in output of over 20 per cent are now being reported by countries that did impose lockdowns. The French National Institute of Statistics and Economic Studies (INSEE) recently reported, that the French economy was now operating 21 per cent below pre-crisis levels, a rise from the 32 per cent below previous levels during confinement.

Costing approach for JobKeeper
The JobKeeper program was deliberately designed as a demand driven program. This means the extent of the program would flex in response to the need for the program.

An overriding policy consideration when designing the scheme was to ensure that the level of support for individuals and firms would be available in the quickest available time given the unprecedented nature of the economic shock.

As is the case for any demand driven program, the level of uncertainty around the individual program’s costing is larger than other program designs. This is particularly the case for a new program and even more so when the background economic conditions are so unavoidably uncertain.

When Treasury costed the JobKeeper program two methods were used.

Both methods arrived at a broadly similar level of potential employee coverage, resulting in an estimated total cost of $130 billion and an estimated 6.5 million workers supported by the scheme.

Both methods of exploring the take-up of the JobKeeper payment relied on an assessment of the overall size of the economic shock by industry from both the social distancing measures and the behavioural impact of firms and consumers.

As we had no experience of the economic implications of pandemics beyond theoretical exercises, the overall impact assessment was highly uncertain.

Moreover, as the JobKeeper program was a new program, we could not use historical experience to judge the take-up of the program, nor did we have detailed business-by-business estimates of the turnover implications of the economic shock.

We did assess that given the design of the program and the size of the subsidy, businesses had a strong incentive to participate in the program.

Subsequent information
Treasury continued to actively monitor the state of the economy and its implications for the JobKeeper program throughout the subsequent period.

On 3 April, National Cabinet continued to focus on the central assumption of 6 months of restrictions. However, the level of restrictions were broadly maintained, and there was not a move to tighten further to allow only essential activities. Such a move would have restricted activities such as construction, manufacturing and non-essential retail more broadly. National Cabinet also acknowledged that there would be variations in state jurisdictions depending on their individual circumstances.

In the following period, there was a reduction in the growth of new cases, with the 5 day average case increase falling relatively rapidly from 388 on 29 March to 238 after 1 week (4 April) and 101 after two weeks (11 April). We know that the health led approach in Australia has been highly successful such that we today enjoy extremely low infection rates albeit in circumstances where continued public health vigilance is required.

As decisions evolved around the degree of economic restrictions, assessments of the likely severity of the economic contraction also evolved. On 9 April, as it became clearer that Australia may not need to tighten restrictions further, briefing was provided to the Treasurer outlining that a central expectation was for a GDP fall in the June quarter in the order of 10 per cent and for the unemployment rate to peak at around 10 per cent in the June quarter.

This was in line with subsequent announcements by other forecasters. The RBA Governor announced in a speech on 21 April that “output was likely to fall by 10 per cent over the first half of 2020, with most of this decline taking place in the June quarter”.
We also began to receive information about the economic impact, from data and through the business liaison program. Notably, the ABS’s early release of single touch payroll data on 21 April indicated an overall decrease in total wages paid of 6.7 per cent and employee jobs of 6.0 per cent between the weeks ending 14 March and 4 April.6 The April labour force release also showed that the unemployment rate was close to the 10 per cent that had been expected, albeit with a large share of people not counted as unemployed in a technical sense.
National Cabinet subsequently set out a phased plan to ease restrictions on 8 May, significantly earlier than initially expected.

The economic outlook will continue to evolve. We now know that containment measures are being lifted more quickly than previously assumed. And we continually receive data on economic conditions to inform the outlook.

Roll out of the JobKeeper program
The Jobkeeper program opened for enrolment on 20 April, following strong interest from business through the ATO register of interest. Enrolments increased steadily from that period onwards, as did the indications of the number of employees covered.

Despite the economic outlook not being as severe as assumed in the costing, the reported JobKeeper enrolments data from the ATO tracked steadily towards the original costings estimate. Treasury considered that this could reflect variations in distributional impacts of the shock across firms, a higher proportion of firms than we had expected being able to demonstrate their eligibility, or the shock being more significant than other real time indicators were implying.

Employee declarations for JobKeeper were submitted from 4 May and JobKeeper support payments started to flow to businesses on 6 May.

The ATO administration process involved two-steps. First businesses had to enrol in the program after assessing they met the turnover test. At this point, they provided an estimate to the ATO of the number of eligible employees they were likely to have. Second, businesses had to make a declaration to the ATO about the actual number of eligible employees they had, including their names and tax file numbers. This two-step process was designed to support the rapid and accurate roll out of the program, with the focus on ensuring accurate information about the firms who would be subsequently receiving the payment from the Government.

The ATO systems have shown themselves to be stable while delivering all required elements of the program on time. The ATO has also effectively delivered other elements of the Government’s response to the COVID-19 crisis, including the cash flow boost measure and early release of super measure.

Once actual payments of JobKeeper began to be made, they flowed at a slower rate than expected. In part, this was assessed as reflecting delays in the finalisation of some rules and guidance material which were particularly relevant for larger employers.
On 13 May, Treasury engaged with the ATO to further understand why payments appeared to be increasing at a slower rate than earlier indications had suggested. It was identified that a high proportion of early declarations were being made by firms with small numbers of employees.

The ATO contacted a number of larger employers who explained that in some cases, it was taking them longer than expected to get the required information back from their employees. In other cases, employers were still assessing their eligibility or undertaking internal governance processes to finalise their application.

After further engagement and analysis, the ATO identified the mis‑reporting error and Treasury was informed and analysed the impact on the afternoon of 21 May. This information was conveyed to the Chief of Staff to the Treasurer in the early evening of 21 May and the Treasurer later that evening.

The enrolment data which were the subject of the reporting error, were collected to provide an early estimate of the number of expected employees likely to access the JobKeeper program. They were also collected to assist the ATO prepare for the payment phase of the program, including ensuring their IT systems would be able to handle the expected load. These data were collected from employers via the enrolment form, and simply required an employer to enter the number of eligible employees who would be remunerated $1500 or more in each of the JobKeeper fortnights in April. There have been errors identified in only around 0.1 per cent of enrolment forms. That over 99 per cent of forms were correctly completed, suggests the form was well-designed.

With the benefit of hindsight, closer analysis of this employee data taken from the enrolment forms should have been undertaken. However, given that the information that was being collected from these forms was lining up with earlier estimates of take-up under the program, and given that payments made under the program are not based on enrolment data, it was decided that resources should instead focus on the payments being made under the program.

This reporting error had no implications for any payments made under the program. The payments depend on the subsequent declaration that an eligible business makes in relation to each and every eligible employee.

While this reporting error did not have a direct impact on the expected cost of the scheme, it provided further confirmation that the economic effects of the crisis, particularly on the labour market, were more in line with the 9 April labour market assessment than the original ‘worst case’ scenario.

Combined with a greater understanding of the translation from enrolment data to expected payments, the Treasury recommended to the Government that the JobKeeper program be re-costed at around $70 billion, with coverage of 3.5 million workers.

Conclusion
The JobKeeper program has been successful in delivering an unprecedented level of support to Australian workers and firms during a time of unprecedented economic need. It is supporting workers’ incomes, business viability and ensuring there is an ongoing link between workers and their firms.

The first two elements are important for supporting firms and workers through the initial response to the virus.

The ongoing employee-employer link is important to ensure that the economic recovery phase is lower cost, as it will be quicker and less costly for firms to expand their labour use as the restrictions on activity are eased and as demand returns to the economy. It will also smooth the path back into work for many employees who may otherwise have lost their jobs.

The updated costings of the program which reflect that there is significantly less demand for this program than initially expected, do not alter these important elements of the program.

1 - Estimates of Australian effective reproductive numbers were first published on 16 April 2020, available at https://www.health.gov.au/resources/publications/modelling-the-current-impact-of-covid-19-in-australia.

2 - COVID-19 Daily Cases, sourced from the Department of Health. Includes all daily cases reported over a 24 hour period. Data can be retrieved here: https://www.health.gov.au/news/health-alerts/novel-coronavirus-2019-ncov-health-alert/coronavirus-covid-19-current-situation-and-case-numbers

3 - See pg 10, Figure 2, of the draft manuscript published by the Doherty Institute on 7 April, available at: https://www.doherty.edu.au/uploads/content_doc/McVernon_Modelling_COVID-19_07Apr1_with_appendix.pdf.  As outlined by the Commonwealth Chief Medical Officer in his Press Conference of 7 April, this modelling was not based on Australian transmission rates.

4 - The advice said ‘less employed persons working’.

5 - The advice said ‘less employed persons working’.

6 - Subsequently, the 5 May release indicated an overall decrease in the total wages paid of 8.2 per cent and employee jobs of 7.5 per cent between the weeks ending 14 March and 18 April. The release on 19 May indicated an overall decrease in the total wages paid of 5.4 per cent and employee jobs of 7.3 per cent between the weeks ending 14 March and 2 May.
'''

SUMMARISER = 1.1


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.

    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences.
        To solve this, we're dividing every sentence score by the number of words in the sentence.

        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    '''
    We already have a sentence tokenizer, so we just need
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, SUMMARISER * threshold)

    return summary


if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)
