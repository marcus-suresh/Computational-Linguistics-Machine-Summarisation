from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

text_str = '''
International Economic Developments
Members commenced their discussion of the global economy by noting that many countries had seen a reduction in the number of new COVID-19 cases and had begun easing some restrictions on activity. The scale of the initial impact on economic activity in many of these economies had also become clearer in the available data. In contrast, a number of emerging market and low-income economies had yet to see a peak in the flow of new cases. Members observed that there continued to be material downside risks to the global outlook, especially if infection rates were to rise in countries where the epidemic had seemed to be under control, or if concerns about the virus or insufficient policy support were to constrain spending.

Measures of core inflation had eased in most advanced economies in April, and lower oil prices had also reduced headline inflation significantly. The effects on prices of the shortfall in global demand and the increase in spare capacity in labour markets had outweighed any boost to prices from disruptions to supply chains.

China had been affected by the virus and responded with restrictions earlier than other countries, and was therefore further into the recovery phase. Industrial production had recovered to be above pre-outbreak levels and measures of traffic congestion had reached previous levels. In contrast, consumer spending and indicators of public transport usage had remained weak, even though most restrictions on movement had been lifted. Members noted that it was unclear whether this weakness reflected ongoing concerns about the virus, a loss of income during the lockdown or more enduring behavioural changes. The Chinese authorities had announced further stimulus measures at the National People's Congress in May, but the scale of the overall stimulus was still expected to be smaller than during the global financial crisis.

Exports from China had also returned to more typical levels by April, but at least some of this recovery appeared to have reflected exporters catching up with their backlog of orders. External demand was expected to be weaker in coming months because the pandemic had continued to constrain demand elsewhere in the world. Consistent with weak global demand, business conditions were reported to be weak in a number of other economies in east Asia, despite the flow of new COVID-19 cases declining to very low levels and restrictions in those economies having been eased substantially.

The major advanced economies had recorded significant contractions in output in the March quarter, even though restrictions on activity had been imposed only late in the quarter. More timely data on business conditions and the movement of people suggested that these economies had been weakest in April and had started to recover in May. Members noted that measures of the movement of people had tended to decline ahead of, or in some cases even without, explicit restrictions being imposed. This suggested that voluntary decisions by households and businesses had been an important factor. Exceptional fiscal and monetary policy support had been deployed throughout this period and further support was being considered in some of the major advanced economies.

The contraction in activity in the major advanced economies had been accompanied by a very severe deterioration in the labour market. The exact effect on reported unemployment rates had varied widely across countries, depending on the severity of restrictions as well as the scope of policies, such as wage and employment subsidy schemes, designed to preserve employment and prevent layoffs; definitional differences regarding whether employees who had been stood down without pay were classified as unemployed had also had a bearing.

The number of COVID-19 cases had continued to increase in a number of emerging market economies, including India and Brazil. At the beginning of June, Brazil had the second-highest number of confirmed COVID-19 cases after the United States. In India, the growth momentum had not been strong when the global pandemic was declared, although output had increased in the March quarter. The severe restrictions put in place to prevent the spread of COVID-19 had led to sharp declines in activity in the June quarter and a significant increase in unemployment to 20–25 per cent. Given the limited social infrastructure available to support unemployed workers in India, authorities had begun to ease restrictions in May, even as the number of new cases continued to increase.

Consistent with the decline in global demand, prices for energy-related commodities, including oil, liquefied natural gas and thermal coal, had remained low. In contrast, prices for iron ore had increased over the prior month, reflecting expectations of increased infrastructure spending in China and elsewhere. Members noted that the increase in iron ore prices had also been driven by supply constraints in Brazil. Demand for coking coal – another input into steel-making – had not been as strong, partly because India's lockdown had reduced an important source of demand for Australia's coking coal exports.

Domestic Economic Developments
Economic activity in Australia had contracted very significantly in late March and April, but more recent data had suggested that it had begun to recover over the course of May. The March quarter national accounts, which would be released the day after the meeting, were expected to report that growth in output had been slightly negative, which would be a stronger result than in most other advanced economies.

While some categories of discretionary household spending had contracted in March, members noted that the large increase in retail spending was expected to have partly offset this, in contrast to the experience of many other countries. Preliminary data indicated that retail sales had since fallen sharply in April. This was consistent with a range of data sources, including liaison reports, which suggested that household spending had declined sharply in April but had recovered somewhat in May.

The contraction in spending in late March and April had been accompanied by significant job losses, with total hours worked falling by 9 per cent in April. Timelier payroll data suggested that the pace of job losses had slowed towards the end of April. In some of the industries that had been most affected by the restrictions on activity, the number of jobs had stabilised or increased a little, suggesting that the total decline in hours worked may be less than had previously been feared. Nevertheless, members noted that, while this was a welcome development, it still constituted a very significant deterioration. Job losses had continued in other industries, which pointed to the possibility of a more persistent reduction in overall demand than had previously been expected. Information from liaison with firms had highlighted that the pipeline of work was diminishing as firms deferred or cancelled investment and new projects, including in construction and business services. Business conditions in April had remained well below average.

In April, an unusually high proportion of people who had lost their jobs did not actively search for another job, so they had been recorded as having left the labour force and, as a result, the unemployment rate had increased by less than expected to 6.2 per cent. The share of workers recorded as being employed but working zero hours had also increased sharply in April; while some of these workers were likely to have been supported through the JobKeeper program, others would have been stood down without pay and may have become unemployed since then.

While job losses and reduced hours for those still in work had reduced labour income, government support had provided a considerable offset. Households that were already receiving welfare payments had received additional payments, and the JobKeeper program and increased JobSeeker payments had supported incomes for others. In some instances, households had received more income than usual. Lower fuel prices, free childcare and other initiatives, as well as more limited consumption opportunities during the period of restrictions, had also reduced living expenses for many households. The implied increase in household savings had been consistent with the recent large reductions in personal credit and increases in mortgage payments.

Preliminary indicators suggested that business investment was likely to have been flat in the March quarter. Data on capital expenditure intentions had pointed to a substantial and broad-based decline in non-mining investment over the following year or so. Members noted that uncertainty about how the economy would evolve over the remainder of the year had contributed to decisions by businesses to defer investment and was consistent with information from the Bank's liaison program. Members also noted that some state governments had been seeking ways to ease planning restrictions to boost investment. Mining firms' investment intentions had been revised down for both 2019/20 and 2020/21, but the data still suggested that mining investment was past its trough.

Some indicators suggested conditions in established housing markets had stabilised in recent weeks, although turnover remained low and housing prices had fallen in some parts of the country. Information from contacts in the Bank's business liaison program had reported that buyer interest in new dwellings had declined significantly over recent months as uncertainty about incomes and the availability of finance had escalated. An increase in the supply of rental properties and the decline in population growth associated with the closure of borders had contributed to a decline in advertised rents. Some existing renters had also received rent reductions. Members noted that these conditions had dampened the outlook for dwelling investment once the current pipeline of work had been completed.

Public demand had grown strongly in the March quarter. Members noted that the Australian Government and state governments had committed to higher spending across a range of programs, including the JobKeeper program; these measures were expected to deliver significant support to the economy.

International Financial Markets
Members noted that financial conditions globally had been broadly stable or eased a little further over the preceding month. The improvement in market functioning had allowed a number of central banks to scale back their purchases of government bonds. The improvement in financial conditions since the acute market strains in March had been consistent with central bank and fiscal policy measures, a decline in COVID-19 infections and the associated easing of restrictions on economic activity. Central banks had announced a few new measures to ease financial conditions in recent weeks, but had generally shifted their focus to refining existing programs or putting already announced programs into operation. Market participants had not ruled out the prospect of additional monetary policy support in the period ahead, including the possibility of negative policy rates in a number of jurisdictions. However, senior officials from the Federal Reserve indicated that they were not considering negative policy rates in the United States, where such policies were likely to be counterproductive. In China, financial conditions remained supportive, as reflected in the recent pick-up in bank credit growth, assisted by the targeted policies of the People's Bank of China.

Members discussed the sharp recovery in the prices of risky assets since their lows earlier in the year and whether this was warranted given the large decline in global economic activity and the highly uncertain outlook. After falling by over 30 per cent in late February and mid March, major market equity indexes had recovered to be about 10–20 per cent below their recent peaks. The ASX 200 had recovered about half of its earlier fall. Market analysts had projected a more favourable outlook for earnings than for overall economic activity, and equity prices had been supported by lower interest rates. Members noted that various asset purchase programs and backstop facilities put in place by central banks, including those of the Federal Reserve, had supported investor demand for corporate securities, with investment grade corporate bond yields declining to around pre-pandemic levels despite record bond issuance. For lower-quality borrowers, corporate funding conditions remained more challenging.

In sovereign bond markets, yields had been little changed around historical lows in most major economies despite rising issuance as governments borrowed to meet spending commitments. Members noted that yield curves in Australia and the United States had broadly similar shapes, being relatively flat out to three years and upward sloping at longer maturities, although US yields had been lower than those in Australia. In emerging markets, members noted that conditions in sovereign borrowing markets had improved a little further, with policy easing by some central banks supporting bond prices in secondary or primary markets. However, financial conditions remained tighter than prior to the COVID-19 outbreak and concerns remained over a number of countries that had external financing vulnerabilities and/or had struggled to manage the spread of the virus.

In foreign exchange markets, the US dollar had depreciated against most currencies in the recent period. This included most emerging market currencies, although increasing tensions between the United States and China had contributed to a depreciation of both the renminbi and the Hong Kong dollar in the forward market. Members noted that the Australian dollar had appreciated further relative to the US dollar and on a trade-weighted basis, to be around levels previously recorded in January. This had followed the earlier depreciation up to the middle of March, which had been larger than that of most other advanced economy exchange rates. Movements in the Australian dollar over the course of this year had been closely correlated with global equity prices. While commodity prices and interest rate differentials between Australia and other advanced economies had increased in the recent period, they had generally been little changed since earlier in the year.

Domestic Financial Markets
The package of monetary policy measures implemented by the Bank in mid March had contributed to accommodative financing conditions in Australia.

The overnight cash rate had settled over recent weeks at around 13–15 basis points, which was below the target rate but above the deposit rate of 10 basis points for Exchange Settlement balances held at the Bank. This was consistent with large Exchange Settlement balances, which had also reduced activity in the overnight cash market, in line with experience in other countries following substantial increases in central bank balances held by the banking system. On a few days in May, activity had been too low to support the publication of a transaction-based cash rate, and the cash rate had therefore been determined by the Bank using the previously published rate, in line with the fall-back procedures. Market pricing suggested that market participants were expecting the cash rate to remain around current levels for some time. Other domestic money market rates were also at low levels. The amounts of the Bank's daily repurchase operations had remained well below their heightened levels in March, reflecting reduced demand and the high levels of liquidity already in the system.

The functioning of government bond markets had improved, with bid-offer spreads returning to more typical levels. Also, the yield on 3-year Australian Government bonds had remained at the target of around 25 basis points. Accordingly, the Bank had conducted only one auction to purchase bonds since the previous meeting, with total purchases to date of around $50 billion. Members noted, however, that the yields on bonds with one to two years to maturity had risen to be a few basis points higher than the yields on three-year bonds. This reflected the lower level of liquidity in these short-term bonds and the tendency for demand to taper off when bonds are no longer in the basket for the three-year bond futures and as they approach maturity. Should these developments continue, the Bank would consider purchasing bonds in the secondary market to ensure that these short-term yields are consistent with the target for three-year yields. 10-year bond yields had been stable around very low levels. These low yields had prevailed despite record issuance volumes, including the syndication by the Australian Office of Financial Management (AOFM) of $19 billion of a new 10-year bond, which had been met by very strong demand.

Members noted that take-up of the Term Funding Facility (TFF) had been increasing gradually, with total drawdowns to date of around $6 billion out of a total funding allowance of $135 billion. Strong growth in deposits had contributed to this gradual take-up. Nevertheless, in liaison, banks had indicated that they planned to draw upon the TFF in greater volumes over coming months to replace existing debt as it matures. Bank bond issuance had remained subdued in light of the strong growth in bank deposits and the option to use the TFF. Bank funding costs were benefiting from record low interest rates on deposits and wholesale funding sources. Issuance of asset-backed securities had resumed in May, supported by purchases from the AOFM.

Activity in business funding markets had increased. Members noted that bond issuance by non-financial corporations had picked up, with the Board's decision at the previous meeting to broaden the Bank's collateral arrangements for corporate debt viewed as supportive of the market. The volume of equity raised had been high as listed companies bolstered the capital position of their balance sheets. Business credit had been little changed in April. This followed the sharp increase in lending to large businesses in March, which to a large extent had reflected precautionary demand by businesses that had drawn down their existing lines of credit in the face of economic uncertainty. In a number of cases, businesses had left the funds on deposit at their bank. Members noted that bank lending to small and medium-sized businesses had been little changed, reflecting weak demand for credit and a cautious approach to lending by some banks. Interest rates on outstanding business loans had declined further in April.

Housing loan payments (which include payments into offset accounts) had increased sharply in April, consistent with many borrowers saving for precautionary reasons and the curtailment of opportunities for spending. These effects appeared to be larger than the effects of deferred loan payments and reductions in housing loan interest flowing through to borrowers.

Housing loan commitments (excluding refinancing) had been little changed in recent months, despite the slowdown in housing market activity. Members discussed that this may reflect the effect of lags in processing applications lodged before the adverse effect of the pandemic on housing market activity. The refinancing of loans remained high as borrowers sought to take advantage of competition among lenders for borrowers with low credit risk. Housing credit extended to owner-occupiers had slowed only a little and growth in credit to owner-occupiers had remained stable over the six months to April. In contrast, housing credit to investors had declined in April.

Considerations for Monetary Policy
In considering the policy decision, members recognised that the global economy was experiencing a severe downturn as countries sought to contain the COVID-19 outbreak. Labour markets had deteriorated significantly and there had been a sharp rise in unemployment in some economies. Over the preceding month, infection rates had declined in many countries and there had been some easing of restrictions on activity. If this were to continue, a recovery in the global economy could be expected to continue, supported by both the large fiscal packages and the significant easing in monetary policies.

Members recognised that the Australian economy was experiencing the biggest economic contraction since the 1930s. A very large number of people had lost their jobs or were working zero hours, household spending had weakened considerably and some investment was being deferred or had been cancelled. Notwithstanding these developments, it was possible that the downturn would be shallower than earlier expected. The rate of new infections had declined significantly and some restrictions had been eased earlier than had previously been thought likely. However, the outlook remained highly uncertain and the pandemic was likely to have long-lasting effects on the economy.

Members agreed that the Bank's policy package was working broadly as expected. The package had helped to lower funding costs and stabilise financial conditions, and was supporting the economy. The package had also contributed to a significant improvement in the functioning of government bond markets, and the yield on 3-year Australian Government bonds was at the target of around 25 basis points. Given these developments, the Bank had purchased government bonds on only one occasion since the previous meeting, although it was prepared to scale up these purchases again, if necessary, to achieve the yield target and ensure bond markets remain functional. The yield target was expected to remain in place until progress was made towards the goals for full employment and inflation.

The Board recognised that the substantial, coordinated and unprecedented easing of fiscal and monetary policy in Australia was helping the economy through this difficult period. It was likely that this fiscal and monetary support would be required for some time. The Board remained committed to supporting jobs, incomes and businesses and to making sure that Australia is well placed for recovery. Its actions were keeping funding costs low and supporting the supply of credit to households and businesses. This accommodative approach would be maintained as long as required.

The Decision
The Board reaffirmed the elements of the policy package announced on 19 March 2020, namely:

a target for the cash rate of 0.25 per cent
a target of 0.25 per cent for the yield on 3-year Australian Government bonds
the Term Funding Facility to support credit to businesses, particularly small and medium-sized businesses
an interest rate of 10 basis points on Exchange Settlement balances held by financial institutions at the Bank.
The Board affirmed that the target for three-year yields would be maintained until progress is made towards the Bank's goals of full employment and the inflation target, and that it would be appropriate to remove the yield target before the cash rate itself is raised. The Board determined that it would not increase the cash rate target until progress is made towards full employment and it is confident that inflation will be sustainably within the 2–3 per cent target band.
'''


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
    summary = _generate_summary(sentences, sentence_scores, 1.4 * threshold)

    return summary


if __name__ == '__main__':
    result = run_summarization(text_str)
    print(result)
