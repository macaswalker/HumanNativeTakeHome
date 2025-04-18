# Take-Home Assignment

## First Look Notes

Aurora Articles: They publish blogs and articles (hence, heavily text based)

Some articles contain personally identifiable information (we need to define what this is)

We want to build a service that allows licensees to report data the expect is
in violation of local laws or regulations

The service will allow consumers to provide structured information about
WHERE: where in the original piece of media
WHY: what it is violation of
HOW: how it is in violation of local laws and regulation

This system supports all types of multimedia (text, image, audio, video, animation)

Currently, we have a team of people in operations that review the manually submitted violations and mark them as valid or invalid. They receive nearly 100,00 violations a month and this number is growing - we NEED automate.

The engineering team is looking for feedback on their data model to see whether the way they store their data makes the machine learning task easy or hard, but have decided to store flagged data as follows:

Dataset(ord_id, id, name, type)

Data(dataset_id, id, value, flag)

We want to build a model that can automatically flag data that might be in violation.

So what is the problem?

We want something that takes in a set of text (at its MVP level), and returns a dataset. we want this dataset 


First Thoughs - Getting an API to do it!

Average Blog Post: 1500-2000 words, 
7500 - 13000 - lets say 10000 characters.

We have 300,000 pieces of media already! 

Assuming we are only working with textual data - this is 
300,000 * 10000 = 3000000000 or 3 Billion Characthers!

Hence, if we are going to use APIs, we cannot use anything prohibitively expensive.



Microsoft Azure: Measure in Text Records (1000 characters), hence 3million text records - £1060.835 pounds per month )-: (https://azure.microsoft.com/en-us/pricing/details/cognitive-services/language-service/)

Microsoft Presidio - is free! awesome (https://microsoft.github.io/presidio/samples/python/presidio_notebook/)

