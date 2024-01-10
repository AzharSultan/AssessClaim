# AssessClaim

AssessClaim parses patient claim for a colonoscopy procedure (45378). It:
* takes a medical record file in docx format
* identifies key criteria in the record like patient age, treatment status, symptoms etc
* traverses a decision tree like structure to make a decision based on criteria
* outputs a csv file with value, evidence and confidence for each criteria
* csv file also contains an 'OUTCOME' row which has the decision on the claim, along with reason for the decision and confidence in the decision
* Confidence scores are coded as below:

| Score | Interpretation |
| ----- | -------------- |
| > 0 | Reasonably Confident |
| 0   | Somewhat Confident |
| -1  | Manually Review |

## Requirements
* Machine with docker installed and access to internet
* OpenAI account with access to GPT-4

## Usage Instructions
* Open the pdf file in MS Word and save it as a `.docx` file. This can later be scripted or replaced with alternative pdf2text version that provides a similar quality
* Pull the docker image:
```bash
docker pull azhsultan/assessclaim:latest
```
* Run the script:
```bash
docker run --rm -e OPENAI_API_KEY=<openai-key> -v <local-folder>:/data cohelm:latest <input-path>.docx <output>.csv
```

Note: The docker image is built on apple silicon. In case it does not work on another machine, you can build the image by cloning this repo and running:
```bash
docker build -t assessclaim:latest .
```

## Pending Tasks
- [ ] add final llm check for decision based on extracted medical history
- [ ] add additional llm verifications for prior treatement as it is the most critical & most likely point of failure
- [ ] update age extraction to extract full date of birth and compare against date of record
- [ ] update format in which LLM delivers its answers
- [ ] few shot prompting for prior treatment

## Design Philosphy
LLMs are an impressive tool and we are still discovering the extent of their performance. Their outputs exceed expectations so often that we tend to ignore the occasional mistakes they make. For example, it is entirely possible to feed the patient history and claim guidelines to the LLM and ask it to make a decision on the claim. With some clever prompt engineering, we could get a decent performance as well. 

But what happens when we scale it to process 1000s of claims a day? An accuracy of 98% is not enough anymore because it still means 10s of wrong decisions everyday. Similar to autonomous driving problem, we will come across the long tail of data and no amount of clever prompting will solve it. This has been the curse of AI based systems throughout the last decade.

Solution? Meticulous Engineering. Breakdown the problem into smallest possible pieces and identify which pieces of information need to be extracted with LLM. Then, verify and reverify the pieces of information by prompting and probing in different ways (https://arxiv.org/abs/2309.11495). Use retrieval models to see if you can match what LLM says against the document. LLMs are weak at math, avoid making them do mathematical gymnastics as much as possible. Python can do perfect math, use that. LLMs falter with reasoning from time to time. Reduce the amount of reasoning as much as possible. LLMs can interpret complex statements differently when faced with ambiguity. So, keep the statements simple. KISS (Keep it simple, stupid). Last but not the least, do not engineer your solution for unseen problems. Just collect more data, measure everything and decide what are the most pressing problems based on the data.

All in all, the toy solution in this repo is a framework on how to approach the problem. It is probably not as generalized as clever prompting and letting LLM do the haevy lifting, on this toy scale. But this framework will scale better and it will be possible to fine tune it into a product.

