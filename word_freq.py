from pydoc import doc
import nltk
#nltk.download()
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

#Insert Text
text_str = '''

****************
International economic developments
Members commenced their discussion of international economic developments by noting that the global economic recovery was well under way and that risks to the outlook had become more evenly balanced. Policy settings remained highly accommodative. The rollout of vaccinations had been progressing well in many advanced economies and some large emerging market economies, although in some countries progress had been hindered by limited supplies, logistical issues and vaccine hesitancy. As vaccinations had increased and hospitalisations remained at low levels, population mobility had recovered considerably in many countries, including in the North Atlantic, where stringent restrictions that had been in place earlier in the year had been relaxed.

This relaxation of containment measures had expanded opportunities for the consumption of discretionary services, including contact-intensive activities such as dining out and domestic air travel. There had been a tentative restart to international tourism in some countries, particularly within Europe. Global goods trade and surveyed conditions in the global manufacturing sector had remained strong because of ongoing strength in the consumption of goods in advanced economies. The high level of new export orders, along with backlogs of existing orders that were still to be filled, indicated that the pipeline of work would continue to support global production and goods trade in the months ahead. Delivery times had lengthened further and global shipping costs were very high.

Members noted that, despite these positive developments, the global economic recovery remained uneven. Ongoing outbreaks of the virus, including of more transmissible variants, had hampered the recovery in many economies. A significant resurgence in cases in Asia over preceding months was contributing to ongoing complications in global supply chains.

Consistent with the global supply chain pressures and higher commodity prices, upstream producer prices had continued to increase in many countries. This was more apparent for goods than services, although producer prices were generally increasing at a faster pace than downstream consumer prices. Headline consumer price inflation had increased in advanced economies, although price pressures were generally narrowly concentrated and expected to be transitory; they were also less evident in most measures of underlying inflation. Medium-term inflation expectations had increased in many advanced economies, but only to rates consistent with, or a little below, central banks' targets.

Labour markets had continued to recover in the major advanced economies and leading indicators of labour demand, such as job ads and employment intentions, remained strong. However, employment and hours worked generally remained below pre-pandemic levels, in contrast to the position in Australia. Labour force participation rates were also still lower than before the pandemic in most advanced economies. Health concerns, ongoing school closures and considerable household income support had contributed to this. In countries where the recovery in labour supply had lagged behind labour demand, pockets of wage pressures had emerged. The prospect of a quick return to tight labour market conditions was most salient for the United States, where substantial fiscal stimulus continued to support demand.

Australia's terms of trade were expected to have reached near record levels in the June quarter, led by high iron ore prices and a sharp rebound in the price of coal. Strong underlying demand from China and steel producers in other countries had continued to underpin high iron ore prices, although prices had been volatile of late in response to intermittent supply concerns and efforts of Chinese authorities to contain prices by dampening speculation. The decline in Chinese imports of Australian coal had been largely offset by a pick-up in demand from elsewhere, and thermal coal prices had surged in recent weeks to their highest level in 10 years, as a warmer-than-expected summer had increased energy demand in parts of Asia.

Domestic economic developments
Turning to domestic economic developments, members noted that the Australian economy was transitioning from recovery to expansion. GDP had increased by a stronger-than-expected 1.8 per cent in the March quarter to be almost 1 per cent above its pre-pandemic level. The solid momentum in growth had continued into the early part of the June quarter. Recent COVID-19 outbreaks in many parts of the country, and associated restrictions, were considered likely to weigh on household consumption through the middle of the year. However, as observed following earlier lockdowns, spending was expected to rebound when containment measures were eased, supported by highly accommodative policy settings, the strengthened balance sheets of many households and firms, and an increase in the pace of vaccinations.

Household consumption had increased by a little over 1 per cent in the March quarter, supported by growth in labour and financial income, and the household saving ratio remained around 12 per cent. Consumption growth in the June quarter as a whole was expected to have moderated, as restrictions on activity in Melbourne and the Greater Sydney region restrained spending. Looking forward, households had scope to increase spending further as restrictions were eased.

Members noted that private investment had been stronger than expected in the March quarter, and that the outlook was positive. Business investment had responded to policy measures, including the temporary full expensing of asset purchases. The outlook for investment was supported by the recovery in demand and high capacity utilisation, the increase in corporate profits over the preceding year and accommodative financing conditions. Surveyed measures of business conditions were also very strong. The near-term outlook for dwelling investment was underpinned by low interest rates and the large pipeline of construction work facilitated by the HomeBuilder program and state-based fiscal programs.

Members discussed the continuing strength in established housing markets. Nationally, housing prices had increased by more than 10 per cent in the first half of the year; a similar pace of growth had been recorded in other advanced economies. Conditions were strong in both capital cities and regional areas, and across different price segments. Over the first half of the year, there had been increases in prices of units alongside further rises in prices of houses. Demand over the preceding year had been led by owner-occupiers, although investors had become more active in recent months. As had been the case for some time, the flow of new listings for sale remained similar to pre-pandemic levels but total listings were much lower, implying that dwellings were being sold at a rapid pace. Rental markets had also tightened noticeably in most parts of the country compared with a year earlier; rents for units in Melbourne had been the main exception, though even there the pace of decline in rents had eased over prior months.

Turning to the labour market, members noted that the unemployment rate had declined further and faster than expected earlier in the year, to pre-pandemic levels. Job vacancies had increased significantly and had reached multi-decade highs as a share of the labour force. Some industries where temporary visa holders had accounted for a sizeable share of employment in prior years, such as hospitality, had high levels of vacancies while employment remained below pre-pandemic levels. This suggested that the closure of Australia's international borders was contributing to the difficulties faced by some firms in attracting workers at prevailing wages. However, new advertised positions were still being filled at a pace consistent with pre-pandemic patterns.

Members held a detailed discussion of the drivers of wages growth over the preceding decade and the factors likely to affect wage outcomes in the period ahead. With indicators of future labour demand remaining strong and the participation rate around record high levels, it was likely that spare capacity in the labour market would continue to decline in subsequent quarters. Consistent with this, members noted that there were some signs of wages growth picking up from the historically low levels recorded following the onset of the pandemic. In many cases, this reflected a recovery to the 2–2½ per cent average annual growth rate seen before the pandemic, rather than a shift to a materially faster pace. Information from the Bank's liaison program suggested that firms were not expecting to make up for the period of wage freezes earlier in the pandemic. Further, the outlook for public sector wages, and rates of wages growth featuring in new enterprise bargaining agreements, suggested aggregate wage outcomes were unlikely to increase materially beyond pre-pandemic rates in coming quarters. Nevertheless, there were ongoing reports of firms using bonuses and other non-wage incentives to attract and retain labour, within a general environment of subdued wages growth.

Members discussed the evidence that wages growth was stronger in local labour markets with an unemployment rate below 5 per cent. They also discussed the timing of the reopening of Australia's international borders and its implications for labour supply. The decline in the number of temporary visa holders in Australia over the preceding year was consistent with reports of tightening labour market conditions in some sectors and regions.

In considering developments in labour supply, members also noted that labour force participation had increased over recent years and that a further increase was likely. This would contribute to available labour supply and supplement the additional hours sought by underemployed workers. Members therefore agreed that, ultimately, a tighter labour market would be required to overcome the inertia in wage- and price-setting norms and the focus on cost control established over much of the prior decade in Australia. Although there were plausible alternative paths for the economy in the period ahead, the central scenario remained one where wages growth and underlying inflation were expected to increase only gradually over the subsequent 2 years.

Members concluded their discussion of the domestic economy by focusing on longer-term developments relating to productivity covered in the recently published 2021 Intergenerational Report. Compared with the 1990s, productivity growth in Australia and other advanced economies had slowed, reflecting a complex interplay of demographic trends, technological change, regulatory developments and other structural shifts. Business dynamism, including the pace of new innovations and their adoption throughout the economy, would be central to lifting Australian living standards over time.

International financial markets
Members commenced their discussion of developments in international financial markets by noting that markets had interpreted the communications from the US Federal Reserve in June as being less dovish than expected. In particular, participants in the Federal Open Market Committee (FOMC) meeting had revised upwards their projections for economic growth and inflation in the near term, and in some cases had brought forward their expected timing of the first increase in the federal funds rate. This led to increases in US policy-rate expectations and shorter-term government bond yields. Longer-term US government bond yields had been volatile in response to the FOMC meeting and had declined over the preceding month, reflecting a decline in longer-term inflation expectations.

Similar moves in sovereign yields had occurred across a number of advanced economies, including Australia. Even so, across advanced economies, central banks had continued to emphasise that near-term inflation pressures were likely to be temporary and that they remained committed to providing significant monetary policy support until there was evidence of sustained progress towards their inflation and employment goals. Most central banks had maintained the pace of their government bond purchases over the preceding month, although some had indicated that a slowing in the pace of purchases was likely in the period ahead. Members observed that central bank holdings of government bonds, including in Australia, were expected to continue to increase as a share of the total stock outstanding for some time. Equity markets had remained around their recent record highs in most major markets, and conditions in corporate bond markets remained highly accommodative. Overall, financial conditions remained very accommodative.

There had been a broad-based appreciation of the US dollar following the FOMC meeting, to be slightly above its level at the start of the year against other major currencies. The Australian dollar correspondingly had depreciated to its lowest level against the US dollar in the current year; in trade-weighted terms, the Australian dollar had also declined to be a little below its level at the start of the year. Members noted that the depreciation had occurred despite commodity prices rising over the course of the year. In part, this had suggested that recent increases in commodity prices were not expected to persist.

Financial conditions in emerging market economies had been relatively stable in preceding months, despite the prospect of rising policy rates in advanced economies and central banks in some emerging market economies having increased policy rates in response to rising inflationary pressures domestically. Financial conditions remained accommodative in China, although credit growth had slowed, consistent with the authorities' goal of managing financial stability risks.

Domestic financial markets
In Australia, the Bank's policy measures had continued to underpin very low interest rates and support the availability of credit. There had been little reaction in financial markets to the recent COVID-19 outbreaks and new restrictions imposed in many parts of the country in late June.

Yields on Australian Government Securities (AGS) with a maturity of 3 to 5 years had risen in June, following the FOMC meeting, the strong domestic labour force release for May and a bring-forward of expectations on the part of some market economists of the first increase in the cash rate. The yield on the April 2024 Australian government bond dipped below 10 basis points for a time in early June, but had risen to be close to 10 basis points subsequently. Prior to the Board meeting, the yield on the November 2024 bond had risen sharply, implying that market participants did not expect the Board to extend the yield target to this next 3-year bond.

Yields on longer-term AGS had declined in June alongside similar moves in US Treasury yields, reflecting lower longer-term inflation expectations in both countries. Members noted that yields for longer-term AGS had been moving broadly in line with US Treasury yields for much of the year. Members also noted that spreads of yields on bonds issued by the states and territories (semis) to AGS had widened a little following the announcement of a larger-than-expected debt funding task for New South Wales, and that yields on semis had been highly correlated across the states and territories. The Bank's bond purchase program had continued to run smoothly and bond markets had continued to function well.

Bank funding costs and lending rates had drifted down to historic lows. At the closure of the draw-down window for the Term Funding Facility (TFF) at the end of June, banks had accessed $188 billion, or almost 90 per cent of the total allowances, with the major banks and mid-sized Australian banks having drawn down all of their allowances. TFF funding had accounted for around 4 per cent of all bank funding. Members noted that banks had also issued some bonds of late, but issuance remained below average.

Demand for housing finance had strengthened further in May, with stronger growth in credit to both owner-occupiers and investors. While bank lending to businesses had been little changed over preceding months, growth in broader measures of business debt had picked up to around the average of prior years. This pick-up had reflected lending to large businesses, including by entities that do not report to the Australian Prudential Regulation Authority as well as through corporate bond issuance. Lending to small businesses had also been little changed, although refinancing activity remained higher than usual as small businesses took advantage of the lower interest rates on offer.

Considerations for monetary policy
In considering the policy decision, members observed that the economic recovery in Australia had been stronger than earlier expected and that this was forecast to continue. Domestic financial conditions remained very supportive and the exchange rate had depreciated a little. Household and business balance sheets were generally in good shape. The effect of the recent virus outbreaks and the lockdowns had created additional uncertainty, but the experience to date had been that the economy bounced back quickly once outbreaks were contained and restrictions eased.

The recovery in the labour market had continued to be faster than expected and more Australians had jobs than before the pandemic. The unemployment rate had declined further. Job vacancies were high and more firms had been reporting shortages of labour.

Members observed that outcomes for the nominal side of the economy had not been as positive. Despite the strong recovery in jobs and reports of labour shortages, inflation and wage outcomes remained subdued. While a pick-up in inflation and wages growth was expected, it was likely to be only gradual and modest. Year-ended CPI inflation was expected to rise to be temporarily above the target in the June quarter owing to the reversal of some COVID-19-related price reductions in the previous year, but would subsequently decline.

In view of the environment of rising housing prices and low interest rates, members continued to emphasise the importance of monitoring trends in housing borrowing and ensuring that lending standards are maintained.

Members discussed the 3-year yield target for government bonds. They noted that, when the 3-year target was introduced in March 2020, the Board had judged that the probability of the cash rate increasing over the subsequent 3 years was extremely low. Since that time, the maturity date for the 3-year bond had been extended from April 2023 to April 2024 and would soon move to November 2024. The Bank's central scenario implied that the conditions for an increase in the cash rate would not be met until 2024. Even so, the faster-than-expected recovery in economic conditions over the course of 2021 had widened the range of alternative plausible scenarios for the economic outlook and therefore the cash rate over the period to November 2024. In view of these considerations, the Board decided to retain the April 2024 bond as the target bond, rather than extend the horizon to the bond with a maturity date of November 2024.

Members also discussed the future of the existing bond purchase program after the second $100 billion tranche of purchases is completed in early September. They noted the previously agreed framework for making decisions about bond purchases. This framework took into account the effectiveness of the bond purchases to date, the decisions of other central banks and, most importantly, progress towards the Board's goals for inflation and employment. Members agreed that the bond purchase program had lowered risk-free rates across the yield curve in Australia, and in doing so had reduced borrowing costs for households and businesses, supported asset values and contributed to a lower exchange rate than would otherwise have been the case. Through these channels, the bond purchase program had been one of the factors underpinning the accommodative conditions necessary for economic recovery from the pandemic. Given the high degree of uncertainty about the economic outlook, members agreed that there should be flexibility to increase or reduce weekly bond purchases in the future, as warranted by the state of the economy at the time, rather than a commitment to a specific rate of purchases over an extended period.

Members then discussed the appropriate pace of future bond purchases. In particular, members considered whether to continue with the current pace of $5 billion per week or reduce purchases to $4 billion per week. Both amounts were within the range of market expectations. Members noted the evidence that central bank bond purchases had their effect through the total stock of bonds purchased, not the flow of those purchases.

Members acknowledged that an argument could be made to retain the pace of bond purchases at $5 billion per week, given that economic outcomes were still well short of the Bank's goals for inflation and employment. However, the economic outcomes had been materially better than earlier expected and the outlook had improved. In light of these improvements and the agreed decision-making framework, members decided to adjust the weekly purchases from $5 billion to $4 billion and agreed to review the rate of purchases at the November 2021 meeting.

The Board remained committed to maintaining highly supportive monetary conditions for a return to full employment in Australia and inflation consistent with the 2 to 3 per cent target. It will not increase the cash rate until actual inflation is sustainably within the target range. The Bank's central scenario for the economy is that this condition will not be met before 2024. Meeting it will require the labour market to be tight enough to generate wages growth that is materially higher than it was at the time of the meeting.

The decision
The Board decided upon the following policy settings:

retain the April 2024 bond as the bond for the yield target and retain the target of 10 basis points
continue purchasing government bonds after the completion of the current bond purchase program in early September 2021 – these purchases will be at the rate of $4 billion per week until at least mid November 2021
maintain the cash rate target at 10 basis points and the interest rate on Exchange Settlement balances of zero per cent.
****************

'''

SUMMARISER = 1.4


def _create_frequency_table(text_string) -> dict:
    """
    Create a dictionary for the word frequency table.
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
    Score a sentence by its words
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
        Long sentences will have an advantage over short sentences.
        To solve this, we divide every sentence score by the number of words in the sentence.

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
    Run the sent_tokenize() method to create the array of sentences.
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


print(text_str)
print(result)

#Post abstraction analysis Part 1 - Sentence scoring#
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("spacytextblob")
doc = nlp(result)
print(doc._.blob.polarity)
print(doc._.blob.subjectivity)

