from pydoc import doc
import nltk
#nltk.download()
#nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

# import spacy
# from spacytextblob.spacytextblob import SpacyTextBlob

#Insert Text
text_str = '''

****************
International economic developments
Members commenced their discussion of international economic developments by observing that inflation abroad had remained high. Inflation outcomes for the major advanced economies had continued mostly to surprise on the upside – noticeably for the euro area, where higher prices for food and energy had boosted headline inflation. Core inflation had remained high in both the United States and the euro area, with services prices increasing strongly in these economies.

Domestic demand had held up in most advanced economies in preceding quarters, reflecting relatively high levels of retail spending and a further transition in spending from goods to services following the pandemic. Nonetheless, the outlook for output growth had deteriorated for most countries, driven by the effects of inflation and higher interest rates on household budgets. In China, economic activity had been adversely affected by further lockdowns in pursuit of the authorities’ approach to managing COVID-19.

Members noted that gas prices in the euro area had declined as storage levels reached capacity and mild weather in the northern hemisphere autumn helped to reduce demand relative to previous years; prices were, however, still above levels prevailing before the pandemic. A similar pattern of price decline had occurred in the market for thermal coal, though the price was still several times pre-pandemic levels. Prices of industrial-related commodities, including base metals and iron ore, had fallen to levels seen at the beginning of 2022.

Labour markets in most advanced economies had remained very tight but had not tightened any further over the preceding month. There were tentative signs of an inflection point in some countries, with falls in employment being recorded recently. By contrast, US employment had continued to grow and unemployment was still very low, although survey data suggested that jobs had become easier to fill in recent months.

Domestic economic developments
Turning to the domestic economy, members observed that economic growth in the September quarter appeared to have been solid, including in household spending. The increase in the value of retail sales in September and other indicators suggested that spending in the September quarter as a whole had risen by around 1½ per cent in real terms.

Members noted the apparent divergence between indicators of consumer sentiment and household spending and discussed the outlook for spending. On the one hand, after a period of catching up following pandemic-related lockdowns and an increase in spending as the year-end approached, growth in spending might slow because of the negative effects of inflation and higher interest rates. While labour income had been rising strongly – underpinned by the tight labour market and solid growth in employment and hours worked in the first half of the year – aggregate real household income had been eroded by inflation and rising interest rates. This was compounded by the effect of falling housing prices on wealth, which would occur over a prolonged period and lead to lower levels of spending on consumer durables. On the other hand, households in aggregate had built up significant savings buffers and it was possible that these would sustain strong growth in consumption in an environment of strong demand for labour.

Bad weather and capacity constraints from supply-chain problems and labour shortages had delayed activity in dwelling and private non-residential construction in the June and September quarters. Capacity constraints continued to affect the outlook; investment plans were delayed in both the private and public sectors. Together with the effect of higher interest rates and continued housing price declines on dwelling investment, ongoing capacity constraints had led to a downgrade in the Bank’s forecasts of these components of investment. Members noted that the outlook for machinery and equipment investment remained positive overall.

The Bank’s central forecast for GDP growth had been revised down a little, with growth of around 3 per cent expected in 2022 and 1½ per cent in 2023 and 2024. The forecast slowdown reflects the combined effects of higher interest rates and lower real wages and wealth on private domestic demand, as well as the broader effects of slower growth in the global economy.

Members discussed the September quarter inflation data, which were a little above the Bank’s forecast. Over the year to the September quarter, the Consumer Price Index (CPI) inflation rate was 7.3 per cent – the highest it had been in more than three decades. In underlying terms, inflation was a little over 6 per cent, with most components of the CPI rising at annualised rates of more than 3 per cent.

Members noted that supply-chain issues and strong demand had continued to boost inflation for new dwellings and consumer durables. However, it was expected that, as supply-chain issues are resolved, declines in transport and other input prices will, over time, flow through to retail prices. Groceries inflation was very high compared with historical data, and the recent floods on the east coast were expected to affect food prices further in coming quarters.

Members discussed the outlook for energy prices, noting the very large forecast increases in electricity and gas prices that had been outlined in the October 2022–23 Australian Government Budget, and which had been factored into the Bank’s revised inflation forecast. Members noted the likelihood of second-round effects on inflation from higher energy prices.

Members discussed the housing market. They noted that rental vacancy rates were low and that many people continue to desire more space than in the past, in part to accommodate working from home. When the international border was closed and the flow of new migrants dried up, rents in Sydney and Melbourne had been quite weak, which had the effect of holding back rent inflation. More recently, rapid growth in advertised rents had led to a pick-up in rent inflation across the country, which was forecast to continue. On the other hand, demand for new detached housing had fallen considerably since the start of 2022; prices for established housing nationally had declined by around 5 per cent since their peak in April.

Following analysis of the September quarter inflation data, the Bank’s central forecast was for CPI inflation to reach 8 per cent by the end of 2022 (revised up from 7¾ per cent previously) and for underlying inflation to be 6½ per cent (revised up from 6 per cent previously). Both headline and underlying inflation were then expected to decline to a little above 3 per cent by the end of 2024 and continue declining in the following year. Higher electricity and gas prices were expected to slow the return of inflation to the target range.

Labour market data for September indicated that employment growth had slowed. The labour market remained tight, with job vacancies and advertisements at very high levels. Members noted that, over preceding months, the strong demand for labour had translated into relatively little additional employment as spare capacity in the labour market had largely been absorbed. The central forecast was for the unemployment rate to remain around 3½ per cent until mid-2023, before increasing to around 4¼ per cent by the end of 2024 as economic growth slows.

Members discussed the review of the Bank’s forecasts over the preceding year. The review focused mainly on the inflation forecasts, with the inflation outcomes having been significantly higher than the Bank and other forecasters had expected a year earlier. In common with other forecasting models used in Australia and abroad, the Bank’s models underestimated inflation. They are not well equipped to deal with large supply shocks and underestimated the impact of global inflation. Changes in firms’ price-setting behaviour had also affected inflation outcomes. Over the year, the inflation forecasts have increasingly incorporated upward adjustments informed by liaison, international experience and specific knowledge of developments in sectors of the economy.

International financial markets
Members observed that financial markets had been volatile over the preceding month, consistent with ongoing uncertainty about the global inflation outlook and the path of policy interest rates. Central banks in most advanced economies had continued the rapid and synchronised tightening of monetary policy.

Over the preceding month, market participants had revised up their expectations for further increases in policy rates in some advanced economies. Most advanced economy central banks had signalled the likelihood that policy rates would be raised further, with some noting that policy rates would need to reach restrictive levels and remain there for some time to return inflation to target. Members noted that some central banks, including the US Federal Reserve, had observed early signs of moderation in growth in demand. Commentary from the European Central Bank and the Bank of England had continued to highlight the risk of recession in their economies, largely due to persistently high energy prices and higher interest rates.

Government bond yields had risen in most advanced economies over the preceding month and volatility had increased notably. In the United Kingdom, government bond yields had been particularly volatile since the government’s announcement of a debt-financed fiscal package in late September. Members noted that UK bond yields had subsequently retraced some of the earlier sharp rise following temporary bond purchases by the Bank of England, announcements that the government would not proceed with most of the fiscal stimulus measures in the mini budget and the appointment of a new Prime Minister.

Private sector financing conditions had tightened further in most advanced economies. Corporate bond yields and credit spreads had risen, although equity prices in most major markets had changed little.

Members noted that the US dollar had appreciated considerably over 2022, particularly against the Japanese yen. Several Asian central banks had intervened in their foreign exchange markets in response to depreciation of their currencies against the US dollar. The Australian dollar had depreciated over prior months but was little changed on a trade-weighted basis over 2022.

In China, volatility in financial markets had increased following the conclusion of the National Congress meeting in October. Equity prices had declined sharply and the Chinese renminbi had depreciated further against the US dollar to around its lowest level since 2007. Property developers remained under significant stress and authorities had announced further targeted measures to support the sector. In most emerging market economies, central banks had increased policy rates further in response to high inflation.

Domestic financial markets
Members noted that Australian financial markets had followed global trends, but generally with more moderate moves. This had been particularly evident in yields on Australian Government Securities, with differentials to US Treasuries continuing to decline, consistent with expectations for a lower peak in the policy rate in Australia than in the United States. Near-term expectations for the cash rate, as implied by market pricing, had decreased over the preceding month following the Board’s decision in October to increase the cash rate by 25 basis points, rather than 50 basis points. At the time of the November meeting, market pricing implied that financial market participants applied approximately a 75 per cent likelihood of a 25 basis point increase in the cash rate and a 25 per cent likelihood of a 50 basis point increase. Market pricing implied that the cash rate was expected to be a little above 3 per cent by the end of 2022 and peak slightly above 4 per cent in mid-2023. By contrast, market economists expected the peak in the cash rate to be a little lower than this.

As a result of the increase in home loan interest rates that had already occurred over the year, housing mortgage payments were set to rise further in the period ahead. This included the effect of fixed interest rate loans rolling off over time. Members noted that, given the cumulative increase in interest rates prior to the November meeting, scheduled housing mortgage payments as a share of household income were expected to increase to levels not seen since around 2010. Payments into offset and redraw accounts were still high, but somewhat less over 2022 than the preceding year. Housing loan commitments had declined further for both owner-occupiers and investors, reflecting the effect of monetary policy on housing lending. The fall to date in housing loan commitments had been broadly in line with historical responses to increases in the cash rate.

Review of the RBA’s approach to forward guidance
Members reviewed the use of forward guidance regarding the cash rate over the COVID-19 pandemic period and discussed the approach to its use in the future. The discussion was based on a paper commissioned by the Board.

Members observed that, prior to the pandemic, the Board had used forward guidance to varying degrees, with the guidance generally being qualitative in nature. However, during the pandemic, forward guidance became more specific and prominent as part of the package of monetary policy responses. Since raising the cash rate earlier in 2022, forward guidance had returned from this more specific form to its earlier more qualitative form.

Members also observed that, together with other monetary policy measures, the Board’s stronger forward guidance had worked to lower funding costs and support the economy in the early stages of the pandemic, when the health and economic outlook appeared dire. The policy response had helped shore up confidence during a period of significant uncertainty and disruption. It had also provided insurance against very bad economic outcomes at a time when there was limited scope to lower the cash rate further.

However, the specific approach to forward guidance had presented substantial communication challenges, which subsequently attracted extensive criticism. The forward guidance had been state-based – that is, with reference to economic conditions – but at various times had included a time-based element.

Members noted that the time-based element of the forward guidance had been prominent in media and market commentary and had come to dominate the interpretation of the Board’s forward guidance. As a result, the Bank had attracted extensive criticism when the cash rate was increased much earlier than the time-based element of the Board’s conditional guidance had suggested. The time-based element of forward guidance and the term for the yield target had been mutually reinforcing. Members noted that the message about the likely timing of future cash rate increases had been complicated by the yield target. The time-based aspect of the forward guidance had not been well suited to the unprecedented global events; moreover, its removal while the yield target remained in place would have significantly affected the credibility of the yield target. A greater emphasis on upside risks to the outlook might have led to an earlier decision to modify the time-based element of forward guidance.

The experience with forward guidance over the pandemic period highlights the communication challenges in combining state-based and time-based elements in forward guidance, particularly with a yield target. The Board is committed to learning from this experience. Many major advanced economy central banks also experienced communication challenges with the combination of state-based and time-based elements in forward guidance through the pandemic. Members noted the public’s interest in understanding the factors that drive the Board’s decisions and that this understanding is an important element in ensuring policy effectiveness and accountability.

Based on its review, the Board decided that its approach to forward guidance will henceforth be based on the following considerations:

Where forward guidance is appropriate, ordinarily it will be qualitative in nature. Given the inherent uncertainty in the world, forward guidance will generally be flexible and conditionality will likely focus on the Board’s policy objectives – namely, inflation and unemployment – rather than the drivers of these variables (e.g. wages). It will typically focus on the short term and be narrative in nature.
Forward guidance on interest rates will not always be provided, although the Board will continue to outline how monetary policy settings will be adjusted in response to evolving economic conditions.
The Board will continue to publish forecasts on a regular basis, along with an assessment of the various risks. The Board does not intend to publish its own forecasts of the expected policy path.
When policy rates are at, or near, the effective lower bound, a stronger form of forward guidance will be considered, taking into account lessons on the benefits of flexibility and using scenarios to prepare for a range of possible outcomes.
Members agreed to publish the review of the RBA’s approach to forward guidance.

Considerations for monetary policy
In considering the policy decision, members noted that inflation in Australia remained too high, as was the case in most countries. Global factors were a key part of the explanation, but strong domestic demand relative to the ability of the economy to meet that demand was contributing to high inflation. Inflation in the September quarter was a little higher than had been expected and had contributed to a modest upward revision of the Bank’s central forecast. So far, medium-term inflation expectations and wages growth remained consistent with inflation returning to target. Members emphasised the importance of this continuing to be the case.

Members noted that the Australian economy had continued to grow solidly. Growth in output was expected to moderate over the following year as the global economy slows, the bounce-back in spending on services runs its course, and growth in household consumption slows in the face of higher inflation and tightening financial conditions.

The labour market remained very tight and many firms were having difficulty hiring workers. Employment growth had slowed over preceding months as spare capacity in the labour market had become limited. The central forecast was for the unemployment rate to increase gradually as economic growth slows.

Wages growth had continued to pick up from the low rates of recent years, although it remained lower than in many other advanced economies. The tight labour market conditions and higher inflation were expected to result in a further pick-up in wages growth. Given the importance of avoiding a price-wage spiral, the Board will continue to pay close attention to both the evolution of the price-setting behaviour of firms and labour costs in the period ahead.

In view of the high current rate of inflation and the forecast for inflation, members agreed that a further increase in the cash rate was necessary to achieve a more sustainable balance of demand and supply in the Australian economy. Price stability is a prerequisite for a strong economy and a sustained period of full employment. The Board’s priority is therefore to return inflation to the 2 to 3 per cent target range over time, while keeping the economy on an even keel. Members saw the path to achieving this balance as a narrow one clouded in uncertainty.

One of these sources of uncertainty is the outlook for the global economy, which had deteriorated over prior months. Another is how household spending in Australia will respond to the tighter financial conditions. The Board recognised that monetary policy operates with a lag and that the full effect of the increase in interest rates was yet to be felt in mortgage payments, consumer confidence was low and housing prices were declining. Working in the other direction, household spending had remained strong and people had been finding jobs, gaining more hours of work and receiving higher wages. Many households had also built up large financial buffers and the saving rate remained higher than prior to the pandemic. Nevertheless, members observed that a reduction in household saving would be needed if recent rates of consumption growth were to be sustained.

Members again considered two options for the size of the increase in the cash rate: a 25 basis point increase or a 50 basis point increase.

The arguments for a 25 basis point increase rested largely on the fact that the cash rate had been increased materially in a short period of time and that there were lags in the operation of policy. While consumption had held up so far, the higher interest rates and high inflation were putting pressure on household budgets at a time when housing prices were also falling. The full effects of higher interest rates were yet to be felt in mortgage payments. However, the tightening of monetary policy was having a clear effect on the housing market, where prices had declined after earlier large increases and the demand for housing loans had fallen. Previous episodes of lower housing prices and turnover had seen a large effect on consumer spending, in part through the wealth channel of transmission.

Members noted that wages growth had not reached levels that would be inconsistent with the inflation target. While future trends were uncertain, wages growth remained below that in a number of other advanced economies. Members also gave weight to the evidence of some easing in global supply-chain issues and a decline in some commodity prices. In addition, increases in policy rates in advanced economies were likely to entail a period of significantly lower output growth, which would reduce global inflationary pressures.

In considering the size of the increase, members also discussed the value of the Board acting in a consistent manner. Having moved by 25 basis points in the previous month, they considered whether the flow of information since then warranted a 50 basis point move at the November meeting. The Board agreed that acting consistently would support confidence in the monetary policy framework among financial market participants and the community more broadly.

As was the case in October, the arguments for an increase of 50 basis points stemmed from the current inflation environment and the upside risks to inflation from the labour market, rents and energy costs. Inflation was at a 30-year high in advanced economies and was broadly based. Any new supply shocks – including in energy markets – could push inflation even higher than forecast in Australia. The tightness of the labour market, with the unemployment rate at its lowest level in almost 50 years, suggested wages growth would pick up further. A risk to the inflation outlook over the medium term was the possibility that price- and wage-setting behaviour would shift, resulting in domestic inflationary pressures becoming more persistent. In their discussion, members noted that many major central banks had been raising policy rates quickly and were more likely to err on the side of doing too much rather than too little. It was also noted that interest rates were still fairly low in a historical context.

Members acknowledged that there were arguments in favour of both courses of action. Given that the cash rate had been increased significantly since May and the full effect of that increase lay ahead, members concluded that the case to increase the cash rate by 25 basis points at the present meeting was the stronger one. Acknowledging the uncertainty, members did not rule out returning to larger increases if the situation warranted. Conversely, the Board is prepared to keep rates unchanged for a period while it assesses the state of the economy and the inflation outlook. Interest rates are not on a pre-set path.

The Board agreed on the importance of returning inflation to target and expects to increase interest rates further over the period ahead in its effort to establish a more sustainable balance of demand and supply in the Australian economy. The Board will continue to monitor the global economy, household spending and price- and wage-setting behaviour closely. The size and timing of future interest rate increases will continue to be determined by the incoming data and the Board’s assessment of the outlook for inflation and the labour market. The Board remains resolute in its determination to return inflation to target and will do what is necessary to achieve that outcome.

The decision
The Board decided to increase the cash rate target by 25 basis points to 2.85 per cent. It also increased the interest rate on Exchange Settlement balances by 25 basis points to 2.75 per cent.
****************

'''

SUMMARISER = 1.45


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


# print(text_str)
# print(result)

# #Post abstraction analysis Part 1 - Sentence scoring#
# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe("spacytextblob")
# doc = nlp(result)
# print(doc._.blob.polarity)
# print(doc._.blob.subjectivity)

