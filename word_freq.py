import nltk
#nltk.download()
#nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

text_str = '''
ntroduction
Hello and thank you for having me here today.

It’s an honour to be able to deliver this year’s 2021 Pearcey Oration, providing some insights into how technology and data are increasingly driving improvements to Australia’s tax system, and how Australia is a world leader in this field.

By way of personal introduction, I have worked in the tax field for almost thirty years, in both the private and public sector, and have witnessed first-hand how advancements in technology have completely transformed the ways Australians manage and interact with their tax and super systems.

As Second Commissioner, Client Engagement Group, my role at the ATO is to 'foster willing participation in the tax and super systems through well-designed client experiences'.

To unpick that corporate jargon a little:

'Willing participation' recognises that the Australian taxation system is world-leading, not because of great auditors (although we have world-leading auditors too), but because the vast majority of Australians are honest, and see the value of their contribution to Australia. That said, they may not always be exuberant or enthusiastic about paying tax, which leads to the next point; which is about, 'Well-designed client experiences'.

This reflects that Australians who are happy to make their contribution should have as simple and painless an experience as possible. We should not make it difficult for them to comply or leave tempting traps where we can help people avoid them. On the other hand, those Australians should also have confidence that others are robustly held to account, and this reinforces their willing participation in the system. We sometimes use the phrase 'easy to comply, but hard not to'.

To achieve these goals, even though Australia is a relatively small country, the ATO must be very interested in digitisation and data. Tonight I hope to give you some insights into how we have approached the challenges and opportunities in our 'core business' of tax collection, as well as how we responded over the last eighteen months to the COVID pandemic and various stimulus measures that we had carriage of. I will also touch on some of our thinking about what is next and where digital and data is going to take us.

Although I am not an IT expert, as a tax administrator, I am fully aware of the challenges of complex client-facing legacy code. For income tax, the core body of our code dates from 1997, but still incorporates significant components of our 1936 code which runs in parallel. For GST, the core body of the code dates back to 1999.

I have split my presentation tonight into several themes:

thinking about the tax return as a data set
insights from the COVID response
some observations on the ethical use of data
how data and digitisation are driving the future of tax administration; and
our part in taking a whole-of-government approach to data and digitisation.
But before I begin, some introductory observations on the Australian tax system and the ATO’s role.

Underpinning the tax system are various Acts of Parliament, and the ATO must administer the tax and super system in accordance with these laws.

One way of thinking about this is; the laws are complex algorithms running inside the ‘ATO computer’ where you theoretically input a factual scenario and out pops the answer about how much tax you owe, and how you must pay it.

These algorithms have been built on over time, with new modules added, patches and fixes to old modules applied, and different programming languages (drafting styles) used. For income tax, there was an attempted algorithm rewrite into a new programming language in 1997, but only part of the system was rewritten and so the old algorithm and new algorithm operate in parallel. The nature of the political process also means that there is limited opportunity for testing prior to going into production, and even less ability to patch the code after defects are identified post-production.

So, although I am not an IT professional, I certainly know a lot about managing buggy legacy code!

You can think of these laws as ‘system specs’ for the ATO and our administration of the tax system, where our goal is to optimise the user interface and hide the unnecessary complexity from users.

Fundamentally, the Australian tax system relies on its users engaging with the ATO as willing participants, as I touched on earlier. I thought it would be worthwhile to quickly give you a quantitative sense of how important this is, how it underpins our strategies, and perhaps dispel the myth that the ATO is all about catching people after the event.

One of the benefits of our data and analytics capability, is that it allows us to put serious effort into quantifying the health of the tax system, both at lodgment, and after compliance activity. We call this ‘tax performance’ or its more glass-half-empty alternative, ‘tax gap’.

Our estimates are that, across the board, tax compliance in Australia, after all ATO compliance activities, is about 93%. That is, we collect 93% of the tax that is theoretically payable according to law. This compares favourably with the very few other countries which have the capability and confidence to measure and publish this information. By way of example, the Commissioner of the Internal Revenue Service recently suggested that the tax performance of the United States was about 80%.

But importantly in Australia, of this 93%, about 90% or just over $400 billion, comes in voluntarily. The ATO compliance activities being the 3% difference with about $15 billion each year. That’s not insignificant, $15 billion pays for a lot of community services, but you can see that it is critical to protect, and grow, the 90% if you want to maintain, or preferably improve, the 93%. It also means that one-off compliance results are not that important. Interventions, which result in sustained future compliance by the taxpayer, are much better.

So yes, the ATO does think extensively about digitisation and data usage in the context of our compliance work, but in most cases, our highest and best use of digitisation and data is in growing the 90% voluntary compliance.

We have made significant investments in strategies to improve our client interfaces, particularly ensuring our technology interfaces are better and easier to use. We are now almost wholly digital. Only about 1.5% of individual tax returns were lodged on paper last year.

We are committed to providing a contemporary, reliable and secure digital experience for all our users – from those who interact with us multiple times a day, to those who only interact with us once a year. We do of course, as a government agency, have to provide services for those who are not able to interact with the tax and super system digitally.

Unlike the select few of us who live and breathe tax, I recognise that for individuals and businesses, tax and super is not always top of mind. In fact, they much prefer when it 'just happens' so we are increasingly looking to:

use insights from client interactions to help design a tax system that makes it easier to comply, and harder not to; and
deliver an integrated and flexible future service capability driven by natural systems and digital events.
While I can’t talk about the intricate details of IT and systems, what I can talk about are the practical implications of using technology and data to drive compliance at the ATO, leading to significant improvements in tax performance.

I can proudly say that the ATO is ahead of the game in that respect because we’ve always dealt in data. Every year, the way we collect, synthesize, use and share data becomes more efficient and effective. The more sophisticated we become with data, the more we know about our world.

I hope from today you are able to gain some valuable insight into how the ATO has leveraged digital infrastructure, data collection and automation to empower businesses and the community, and how we became one of the largest information and communications technology employers in the country.

The tax return as a data set
The tax system pre-dates the ability to obtain data at any scale.

Its underpinning was the tax return, a custom data set that we asked each Australian to manually produce every year. The accuracy of this data set relied upon the honesty of Australians, backstopped by significant penalties and the threat of audit, which was a bespoke and time intensive data verification exercise.

Of course, it was very difficult to verify the information provided in those returns against other data points, even on a one-on-one basis, let alone at scale, because the data was in the possession of the taxpayers.

The next phase of tax administration around the world developed as verifiable third-party data became available to the ATO at scale. This allowed more comprehensive data-matching and the identification of anomalies that require further investigation through audit programs.

The phase we are now in, that many of the leading revenue authorities around the world are also in, is pushing that third-party data to taxpayers to save them time and minimise the risk of inadvertent errors that have to be chased down later.

We now tend to think of data as on a curve:

Level 1 is taxpayer provided data, where there is no bulk data set available, such as work-related expense claims where taxpayers keep their receipts.
Level 2 is where we can obtain data after the event to check that data, but maybe not at scale.
Level 3 is where the data can be sourced to be used as a risk indicator pre or post lodgment but it is not of a quality or type that would be productive to expose to taxpayers.
Level 4 is where the data is of a high enough quality that it can be used to assist taxpayers to comply as they lodge.
Level 5 is where the data is very high quality and can be used to pre-fill returns as presumptively correct.
Level 6 is where the data is so reliable that the tax system is actually designed around the data.
We are increasingly seeing this ‘Level 6’ as the future of tax administration, that is, to move the system to verifiable data, rather than bring the data to the system. The level of data is based on the assessment of six attributes: identity, timeliness, standardisation, reliability, visibility and usefulness. All of this is aimed at making the accurate completion or affirmation of the ‘data set’ or, the tax return, easier for Australians.

We are now at the stage where, for the average individual, about 90% of the boxes we pre-fill on an individual tax return to report income aren’t changed by the taxpayer. This saves Australians a significant amount of time to complete their obligations, whether they complete their returns themselves or via an agent. Not only does prefill include most income items, it also includes other items such as health insurance details. The items which remain, tend to be deductions or relate to more complex matters, such as rental properties and capital gains, however partial data can still assist with those details.

As an example, we get a lot of data from various cryptocurrency exchanges and we are generally aware when a taxpayer has cryptocurrency investments and trades. While we will tend not to have sufficiently high quality data to calculate the exact capital gains, Level 5 on our curve, we do have enough information to inform the taxpayer that we are aware of the fact that they have cryptocurrency trades. We can then remind them that these need to be included in the income tax return (Level 4). A more traditional revenue agency approach would be to remain silent until a return was lodged, and then selectively audit returns where we knew of cryptocurrency holdings but no gains were disclosed (Level 3).

Another example is the good insight we have into what sorts of deductions taxpayers in particular occupations and locations have. We call this our ‘nearest neighbour’ data. We have shifted from using this data as a risk tool to determine who we should audit (Level 3), to actively prompting or ‘nudging’ the taxpayer in real-time when they are completing their tax return. We let them know that their claims are out of the normal range, and give them the opportunity to consider more deeply, prior to final confirmation, whether their claim is valid (Level 4).

To give this strategy a sense of scale, in Tax Time 2020 we issued nearly 350,000 prompts to help taxpayers check their figures in myTax when our systems detected that amounts entered were significantly different to those of others in similar circumstances. The behavioural response to this was interesting. About a third immediately changed their deduction, a third soldiered on, and a third tried to ‘game’ our system by progressively reducing their claim to see when they would get below our risk engine level.

We are getting towards the limit in Australia under current tax policy settings on what we can achieve for individuals. In some countries’ tax systems, these items have been moved to even ‘Level 6’, where the system is designed around the data. For example, through the use of standard deductions, rather than itemised deductions, for work related expenses. Here, in Australia, there are significant policy questions about the possible trade-offs of doing something similar

The COVID response
As I mentioned earlier, the data is not just about tax returns and helping people get their tax and super obligations right. Our response to COVID-19 emphasised the importance of having high-quality systems and data sets in place for more than one purpose. They can come in very handy when the unexpected comes along.

In March last year, the ATO was entrusted to deliver some of the government’s key economic stimulus measures: upwards of $89 billion in JobKeeper, about $35 billion in cash flow boost, and to facilitate the early release of super which, totalled approximately a further $38 billion.

Paying out money is not something that people would traditionally think of as the ATO’s ‘core business’. However, I and the rest of the ATO Executive, are extremely proud of the efforts of our people in the successful delivery of these measures.

Within only a few weeks, we were able to set up systems from scratch and those systems managed to successfully assist 1 million JobKeeper recipients supporting 3.8 million individuals, and 800,000 small businesses receiving cash flow boost. Those systems also facilitated access to superannuation accounts for 3 million Australians. All of this support for the Australian community was achieved with a high level of program integrity, and the work of a very dedicated group of people.

Importantly, in fine-tuning the policy initiatives, the Department of Prime Minister and Cabinet and Treasury worked very closely with the ATO to understand the data holdings and digital systems we already had in place, and designed the policy programs around them.

There were some key data sets which provided the infrastructure for the programs, including:

Australian business numbers for businesses
tax file numbers for employees
previously lodged income tax returns and activity statements; and
Single Touch Payroll (STP).
The policy was designed around these data sets to ensure it was almost impossible to access the programs unless the business was a pre-existing, real business with true employees and with a track record of engagement with the tax system. It also meant that it was easy to link the measures to our data and analytics risk engines.

The other important aspect was to digitise the experience to make it as easy as possible for claims to be made. When we were looking at processing what turned out to be about 5 million forms, that process had to be entirely automated.

Cash flow boost was actually designed so that there were no new claiming obligations. A business simply had to lodge their normal activity statement and our systems would automatically generate the cash flow boost credit.

JobKeeper on the other hand was necessarily more complex, but even then it was fully digitised with no manual processing. The JobKeeper claim process was designed to the maximum extent around the existing myGovID and Single Touch Payroll systems. Here I must call out the digital service providers who rapidly upgraded their STP systems to allow an employer to simply tick a box through the business’ natural system, which links directly to ATO systems.

The outcome of this was that about 97% of JobKeeper claims could be paid within four business days of the ATO processing their monthly claim.

But our goal with JobKeeper wasn’t just about paying out money quickly. We also had to ensure the program had high integrity and provided the community with confidence that their taxpayer dollars were only going to those who were eligible.

I would just say here, that some of the recent commentary around JobKeeper is actually not around eligibility, it’s about whether, with the benefit of hindsight, the particular business was deserving. Eligibility was rock solid.

Our data also allowed us to check eligibility for the assistance payments upfront and in real-time, or soon after application, which meant that fraudulent claims were almost impossible. To give a sense we:

prevented almost 70,000 applications from excluded entities; and
reviewed more than 100,000 entities to a value of about $12 billion.
This equates to about 10% of the JobKeeper population by number and value.

Ultimately, we determined more than 35,000 entities to be ineligible, clawing back payments we had made and stopping future payments of just over $1 billion. I recognise $1 billion is a lot of money but in the context of a scheme of $89 billion, that is a very effective scheme.

Ethical use of data
Growing our data holdings and capabilities has only been possible because of increasing advances in technology. But once you get over the technological constraints, there are also legal constraints and ethical constraints that need to be considered.

Critically, the ATO has been entrusted by the Australian community with extensive information-gathering powers. These powers were granted to help address the information asymmetry between taxpayers and the ATO.

The flip side of these powers is that taxpayers are protected from this information being misused, used by other parts of government unless there is a specific exemption, or made public. A mark of how seriously this is taken by the Australian community is that a breach of taxpayer secrecy is a criminal offence, and one which is potentially subject to custodial sentences. As a tax officer, that certainly does focus the mind on the ethical use of data.

We must reciprocate the trust the community places in us around their data.

For those following the recent press, this is currently playing out in the Senate where there is an order from the Senate for the ATO release of data in relation to JobKeeper payments received by large businesses.

This is a challenging scenario for us. Legally, we must respond to such an Order to Produce documents from the Senate. However, businesses applied for JobKeeper under a set of rules, which provided for taxpayer secrecy. We are strongly arguing that unless the Parliament enacts rules to retrospectively require us to publish the data, or the Senate compels us to release it under current law, it is against public interest to now make this data public. This is not just in relation to this data set, but also for the precedent it sets in relation to other taxpayer information.

Moving beyond the legal framework to ethics, we have developed a set of data ethics principles which are designed to make interactions easier and ensure our data activities support the integrity of Australia’s tax and super systems. Our six principles are to:

act in the public interest, be mindful of the individual
uphold privacy, security and legality
explain clearly and be transparent
engage in purposeful data activities
exercise human supervision; and
maintain data stewardship.
This framework is being supported by the implementation of a ‘data steward’ role, where there is a designated officer responsible for each data set. We are focused on maintaining stewardship over how data is used after it is shared with others.

In addition, we have a robust security framework to ensure the confidentiality and reliability of our digital services, and we continue to adapt and improve our technology to address emerging cyber risks. We are conscious that we sit over one of the most interesting data sets in Australia.

Another element of our data philosophy is to only obtain data where it will be used and its usefulness is proportionate to its burden or intrusiveness.

A good example of how this plays out in practice is e-invoicing, which I’ll discuss in a bit more detail later. In many of the countries that are introducing an e-invoicing type platform, all invoices are effectively routed through the revenue authority. This is intended to identify business to business transactions which otherwise would not be reported to the revenue authorities. Australia however has not gone down this path.

Although the ATO is playing a supporting, or catalysing role, in the implementation of e-invoicing, by encouraging networks on both the supply side and the purchaser side, Australia has recognised that the vast bulk of businesses here operate honestly and accurately report their business to business income to the ATO. Those that do not, are unlikely to start e-invoicing. As such, there is only marginal usefulness to the tax system of running e-invoices through the ATO, and the intrusiveness for business is not proportionate to that.

Looking into the future of tax administration
The exponential growth in the number of digital transactions the ATO processes each year is extraordinary. If we look at Standard Business Reporting (SBR) which was introduced in 2013, we processed 140,000 ‘machine-to-machine’ transactions. In 2020 we topped 1 billion machine-to-machine transactions through SBR. We’re forecasting this to reach over 2 billion in 2022. Similarly, our ‘client interaction’ volumes through ATO Online for individuals, businesses and agents, have increased over 300% since 2018. We understand that this is far more than any private sector organisation.

We’ve bolstered our in-house data analytics capability and have some very talented staff doing clever things with ATO-owned data as well as third-party data. This work sits with me in our client experience area, rather than in our IT areas, which reflects the increasingly integrated way in which data is driving and shaping our engagement approaches. We’re also working closely with digital service providers to make their clients’ data more accessible to them so they can build useful products for businesses.

At a more human level, in the past there were lots of what I sometimes call ‘data sherpa’ roles in the tax system. These were people who picked up information from one place, maybe a spreadsheet or a form or even a shoebox, and took it to another. As technology and data collection and usage has gotten better and better, we’ve seen those roles disappear. We now recognise business models that were shaped around data sherpaing are also under severe pressure.

We have seen these changes impact our own workforce as well as our key partners in the tax profession. Over the past five years, the ATO and the tax profession have been on a journey of digital transformation. We’re continuing to develop digital solutions to make it as easy as possible for tax professionals to manage their clients’ affairs through their own systems to the greatest extent possible; not by having to come onto our systems. This includes real-time analytics and nudges to help them save their clients from making errors.

With a lot of low-value work being taken out of the system, this gives people the opportunity to dedicate more time to doing higher-value work to maximise their profits, rather than doing more administrative tasks.

More sophisticated use of data has also transformed how we can measure our own effectiveness, which in turn flows onto more sophisticated strategies and better operational incentives and metrics.

In the past, revenue authority effectiveness could only be measured by simplistic metrics such as audit yield ratios, for example: how many audit dollars were ‘earned’ for a given level of investment. This was a proxy for tax system health, but a poor one. High audit yield could be a sign of revenue agency effectiveness, but it could also be a sign of a poorly performing system with lots of easy audits. Over time, it is easy to fall into the ‘McNamara fallacy’ which is to measure what is easy to measure, and then over time treat that as the only important thing.

The analogy I draw here is WADA at the Olympics doing drug testing. If they catch a lot of drug cheats, that could be because they are doing a great job catching drug cheats, but it could also be that they are doing a terrible job because there are a lot of drug cheats. So, how do you avoid a metric that encourages simply catching drug-takers, which might actually distract you from actions to prevent drug taking in the first place?

I mentioned earlier the concept of a ‘tax gap’. This is something that has been facilitated through our advancements in data and analytics.

Importantly, we publish our findings each year in our Annual report so that Australians can have visibility of the health of the Australian taxation system, and as I said earlier, we are running at about 93%, which is world-leading performance.

Let me pause here and mention large business. If you read the headlines, you would think that large business tax compliance was in a state of disrepair. We actually think that, based on these metrics, large business tax compliance in Australia is over 93% correct at lodgment and over 96% after our audit activity. So, the gap is actually about $2 billion per year.

While no tax system can eliminate tax gaps completely, we are working towards sustainably reducing tax gaps over time, and we use this program to measure our effectiveness.

Given the already high performance of the Australian tax system, which might be very different to the strategies in other countries with lower performing tax systems, our strategies are focused around protecting the existing levels of high compliance. Only after that do we start to think about how we boost through compliance. Even then with compliance activities, we consider how we ensure compliance in the future. There is no point doing an audit this year, and then having to start again from scratch with an audit the next year.

Another aspect that I think has broad application to large organisations, is around performance measurement. We often hear organisations complain about silo behaviour, where leaders ask, ‘how do I stop siloed behaviour’. My personal perspective is that siloed behaviour is usually occurring for a good reason and it’s probably happening because you have siloed metrics. The performance metrics of your team are encouraging that behaviour and discouraging cross silo collaboration.

Where I think data and analytics can really add to the performance of organisations is by generating more sophisticated performance metrics at a team level, where team leaders can really understand how their team is performing not just on their workbench but on an end-to-end sense. We call this our 'pipeline metrics initiative' and already this is making big differences to how our teams are operating and understand what they do.

Another observation I would like to make is about hearing team leaders complain to the effect: ‘but I tell them they should think across the whole organisation’. To me, it’s the obligation of management to give metrics to team leaders so they don’t have to second-guess the data, so they know that if they follow the metrics they are doing the right thing.

A critical factor in the recent evolution of the tax and super system, and its future, has been the rise of digital service providers (DSPs), and the digitisation of Australian businesses.

The ATO can no longer think of its systems as a tax and super tech island, but rather as part of an ecosystem where our systems interact directly with the natural systems of businesses. As a government agency, we must provide good, direct services to taxpayers. Recent experiences in the US of a market failure in individual tax return software confirms this. Increasingly more of the tax ecosystem involves businesses operating in their natural systems provided by third parties, with that innovation being driven by entrepreneurialism. We must give those third parties the space to innovate and earn a reward for their innovation.

Our immediate focus here is on small businesses. We know that those who are digitally engaged perform better pre-tax and also meet more of their tax and super obligations. By integrating tax and super into natural business systems, we will allow business owners to spend more time on growing, improving and expanding their business, rather than worrying about paperwork and red-tape.

We are currently at the early stages of exploring with business and DSPs whether more of the tax system, particularly relating to small business, can be distributed to help businesses comply and avoid getting into difficulties with the ATO. Some of the questions we are asking ourselves and other participants in the system are:

If we are confident the data is in the natural systems of a business, do we need to ask for it in the form of a tax return or activity statement? Could we rely on a data feed? Do we even need to hold the data at all if we are confident that the tax obligations arising from that data are being met?
How can we help businesses avoid mistakes before lodgment? Can we provide the natural systems access to some of our risk filters? Can this further reduce our need to hold the data underpinning that analysis?
How do we preserve and ensure high quality oversight from tax professionals in a more digital and distributed tax ecosystem?
At the other end of the spectrum, the nature and complexity of large businesses, and the bespoke nature of their systems, means we are still quite some way off from getting to this level of data integration in real-time driving risk analysis or compliance approaches. Instead, we continue to drive more sophisticated analysis of bespoke data sets and are increasingly enabling these taxpayers to undertake their own more complex risk analysis to shape their own behaviour based on ATO or industry wide expectations, rather than driving them to audit.

We are also increasingly looking at automation to drive some of our improvements in how taxpayers can interact with us, and also how we approach our own work.

For example, our virtual assistant, Ask Alex, helps visitors to our website with general enquiries. Ask Alex, had over 1.4 million conversations in Tax Time last year, 94% of which were resolved at first contact.

Other innovations, such as the use of automated responses to simple calls, to assist in some of our information gathering, even in the context of audits, frees up our people so that more of our time can be spent on work requiring human judgement and empathetic support. The past 18 months has really brought home the importance of these human interactions and so, while we embrace technology, one of the benefits it brings is that our people can really focus on those things that only people can do.

Towards a whole-of-government approach
Increasingly, the government is looking at a whole-of-government approach to the provision of services to the Australian community. This of course requires a whole-of-government approach to digitisation and data.

The ATO, as a large and advanced agency, is a key participant in this initiative and a few of the building blocks that we are working on include:

e-invoicing providing a platform for companies to send and receive digitised invoices
Modernising Business Registers, which is improving the quality of government registers, such as the companies register and introducing a Directors register; and
myGovID/RAM, a high-quality identification system for dealing with government.
Each of these initiatives is a critical part of Australia’s future digital infrastructure and will make it easier, cheaper and safer to operate in Australia:

e-invoicing is a platform for business to business invoices and will make it cheaper and less human-intensive for businesses to send, receive and process invoices, as well as reduce payment lags. There is also the possibility of using this underlying PEPPOL framework for other digital communications between businesses.
Modernising Business Registers is about putting the key business-related registers in one place, in one agency, to make it easier to understand what business you are dealing with. In addition, by having a natural touch point with government at the start of a business’s life, it will make it easier to help businesses interact digitally from the outset, as well as being a simple entry point to the range of Government interactions.
myGovID/RAM is a resilient framework for proving your identity, and then also authorising others to deal on your behalf, primarily in a business context but increasingly in an individual context. Already being used across more than 30 agencies, this is ultimately a ‘single set of keys’ for dealing with Government.
The ATO is also playing a leading role in data sharing across government, which will likely only continue to grow under the National Data Sharing Strategy, which is under development, and the proposed Data Availability and Transparency Bill framework. Importantly, the DAT Bill is focused on sharing unit level data for the purposes of enhanced service, not compliance, and so will require strong data stewardship and governance.

From a whole-of-government perspective, and indeed from a citizen perspective, data should ideally only be collected once, and then shared within government, rather than asking citizens to share the same data multiple times across government. This naturally will be with the agency that logically has the greatest relationship with the client’s natural system. Any data collected by the agency through that service can then be shared with other agencies as necessary, subject to legislative constraints.

The perfect example of this is Single Touch Payroll, where the ATO is responsible for collecting large volumes of payroll data from employer’s payroll software and, while this is primarily used for tax purposes, a subset of this data is provided to Services Australia to assist with administering the welfare system. It would be an unnecessary burden on employers for Services Australia in this scenario to require employers to connect and report to them separately.

At a higher level, ATO population level data is increasingly being seen as a whole-of-government asset, critical for understanding what is going on in the economy and informing policy development.

This was evident during the height of the pandemic last year, when we worked with the Australian Bureau of Statistics (ABS) to provide them Single Touch Payroll data, which included accurate payroll information in relation to approximately 92% of the working population in real time. The ABS was then able to use this information, instead of simply relying on its usual employment surveys, to provide high-quality real-time insight to the Government on the state of the economy.

Under this whole-of-government data sharing framework, we are increasingly seeing a few key agencies will act as hubs for data, with that data being made available appropriately for policy development and even research.

Conclusion
Thank you for inviting me to share some insights from the ATO and our experience over the last few years and how, into the future, data and digitisation can improve not just the tax system, but Australian society.

Over the coming years, we hope to continue to further improve, remaining the most digitised revenue authority in the world.

So, when you lodge your next tax return, whether yourself or through an agent, I hope you have a better insight into what we are thinking about behind the scenes, as we continue to strive for a system ‘where tax just happens’.

Thank you.
'''

SUMMARISER = 1.3


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
