__all__ = ["INSTRUCTIONS"]

SUMMARY_INSTRUCTION = (
f"""You're a content writer, and your task is to partly summarize the mian contents of the given ###Context and the two-hop question.

The input contains:
###Context: A title and some sentences.
###Question: A two-hop complex question, in which only one-hop is relevant to the given ###Context.
###Type: question type, is \"bridge\" or \"comparison\". 
- Bridge question is often a clause, we need to found the answer to the main or subordinate part of the ###Question.
- Comparison question is to compare two things, such as time, quantity, etc.
 
You can finish this as follow steps:
1. Find the common contents that the ###Context contains and the ###Question asks.
2. According to the common contents, write a question-oriented summary of the ###Context.
3. Remove the information that is not appears in ###Context.

Here are 2 examples:

##Example 1:
###Context: 
- Summer Magic
- Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, \
and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.

###Question: Which movie was produced first, Summer Magic or Hocus Pocus?
###Type: Comparison

###Reasoning steps: 
1. The common context between the ###Context and the two-hop ###Question is the movie "Summer Magic." 
2. The ###Context provides information about the movie, stating that it is a 1963 Walt Disney Productions film. 
Therefore, we can summarizes this information: "Summer Magic is a Walt Disney Productions film produced in 1963."

###Abstract: Summer Magic is a Walt Disney Productions film produced in 1963.

##Example 2:
###Question: What award was received in 1982 by the professor that supervised Jun Li's Harvard Ph.D?
###Type: Bridge
###BridgeInfo: Shing-Tung Yau

###Supporting Fact Sentences: 
- Jun Li
- He received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau.

###Reasoning steps: 
1. The common context between the ###Context and the two-hop ###Question is the person "Jun Li" and his Ph.D. at Harvard University.
2. The ###Context provides information about Jun Li, stating that he received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau. 
Therefore, we abstract this information by: "Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University."

###Abstract: Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University.

Here is the input information:
###Context: 
{{supporting_facts}}

###Question: {{question}}
###Type: {{question_type}}

Only use the information that's provided in the text.
Tips: if the question type is **Bridge**, the statement is often relevant to the given ###BridgeInfo.
Tell me its abstract and corresponding reasoning steps in json format, with keywords \"valid\", \"reasoning_stesp\" and \"abstract\".
\"valid\" is True if you can answer it, else False.
"""
)

REFINE_INSTRUCTION = (
f"""You're a content writer, and your task is to refine the given summary according to the given context and question.

The input contains:
###Context: A title and some sentences.
###Question: A two-hop complex question, in which only one-hop is relevant to the given ###Context.
###Type: question type, is \"bridge\" or \"comparison\". 
- Bridge question is often a clause, we need to found the answer to the main or subordinate part of the ###Question.
- Comparison question is to compare two things, such as time, quantity, etc.
###Summary: A coarse summary that 1) maybe include content not mentioned in the ###Context, \
2) or is too redundant, which covers far more content than the ###Question, even copies a lot extra words from ###Context.

You can finish this as follow steps:
1. Find the contents that the given ###Question asks in the given Context.
2. Compare it with ###Summary to see if ###Summary has a lot of redundant information.
3. Refine the ###Summary and re-write a  more concise summary.

We provide 2 examples: 

##Example 1:
###Context: 
- Arthur's Magazine
- Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
- Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.

###Question: Which magazine was started first Arthur's Magazine or First for Women?
###Type: comparison

###Summary: Arthur's Magazine was an American literary periodical published in Philadelphia in the 19th century, edited by T.S. Arthur and featuring work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.

###Reasoning steps:
1. The ###Question requires us to find the started time of 2 magazines. And the ###Context provide the information about Arthur's Magazine.
2. Hence we need to found the Arthur's Magazine published time. 
3. In addition to the established time, the ###Summary also contains a lot information about editor and it featured work.
4. Therefore, we can refine the ###Summary by remoiving these redundant words: Arthur's Magazine was an American literary periodical published in 1844.

###Output: Arthur's Magazine was an American literary periodical published in 1844.

##Example 2:
###Context: 
- Boren-McCurdy proposals
- The Boren-McCurdy intelligence reform proposals are two legislative proposals from Senator David Boren and Representative David McCurdy in 1992 (102nd Congress).
- Both pieces of legislation propose the creation of a National Intelligence Director.

###Question: The Boren-McCurdy proposals were partially brought about by which Oklahoma politician in 1992?
###Type: bridge

###Summary: The Boren-McCurdy proposals were two legislative proposals from Senator David Boren and Representative David McCurdy in 1992 that proposed the creation of a National Intelligence Director.

###Reasoning steps:
1. The ###Question asks about the Oklahoma politician who partially brought about the Boren-McCurdy proposals in 1992.
2. The ###Context provides the information that the Boren-McCurdy proposals were two legislative proposals from Senator David Boren and Representative David McCurdy in 1992.
3. The ###Summary already includes this information, but it also mentions the purpose of the proposals, which is the creation of a National Intelligence Director.
4. Since the purpose of the proposals is not relevant to the ###Question, we can refine the ###Summary by removing that part.
5. Therefore, the refined ###Summary would be: The Boren-McCurdy proposals were two legislative proposals from Senator David Boren and Representative David McCurdy in 1992.

###Output: The Boren-McCurdy proposals were two legislative proposals from Senator David Boren and Representative David McCurdy in 1992.

Here is the input information:
###Context: 
{{supporting_facts}}

###Question: {{question}}
###Type: {{question_type}}

###Summary: {{abstract}}

Tell me its output and corresponding reasoning steps in json format, with keywords \"valid\", \"reasoning_stesp\" and \"output\".
\"valid\" is True if you can answer it, else False.
"""   
)

INSTRUCTIONS = {
    "summary": SUMMARY_INSTRUCTION,
    "refine": REFINE_INSTRUCTION
}


if __name__ == "__main__":
    print(SUMMARY_INSTRUCTION)
