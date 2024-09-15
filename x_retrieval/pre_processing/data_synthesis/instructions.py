__all__ = ["INSTRUCTIONS"]

SUMMARY_INSTRUCTION = (
f"""You're a content writer, and your task is to re-write the ###Statement to ###Abstract by removing the contents that ###Context does not mentions.

The input contains:
###Context: A title and its corresponding sentences.
###Statement: A statement but contains the information that is not appears in the , ###Context\
and maybe has some grammar errors. 

You can finish this as follow steps:
1. Find the common contents that both ###Context and ###Statement contain.
2. Remove the noise information that is not appears in ###Context.
3. Fix the grammar error, and keep content consistency.

Here are 2 examples:
##Example1
###Context:
- Jun Li
- He received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau.

###Statement: Fields Medal was received in 1982 by the professor that supervised Jun Li's Harvard Ph. D.

###Reasoning steps:
1. Common contents: Both the context and statement mention Jun Li and his association with Harvard University.
2. Removing irrelevant information: The statement mentions the Fields Medal received in 1982, but this information is not present in the context. Therefore, it is removed from the output.
3. Grammar and content consistency: The output sentence is rephrased to maintain grammatical correctness and content consistency. It states that Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University, which is the relevant information mentioned in the context.

###Output: Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University

##Example2
###Context:
- Bindi Irwin
- She is also known for winning season 21 of \"Dancing with the Stars\" (U.S.).
###Statement: The winner of season 21 of \"Dancing with the Stars\" stars with her brother in a web series that has 24

###Reasoning steps:
1. Common contents: Both the context and statement mention Bindi Irwin and her association with "Dancing with the Stars" (U.S.)
2. Removing irrelevant information: The statement mentions her brother and a web series with 24, but this information is not present in the context. Therefore, it is removed from the output.
3. Grammar and content consistency: The output sentence is rephrased to maintain grammatical correctness and content consistency. It states that Bindi Irwin won season 21 of "Dancing with the Stars" (U.S.), which is the relevant information mentioned in the context.

###Output: Bindi Irwin won season 21 of "Dancing with the Stars" (U.S.)

Here is the input:
###Context: 
{{context}}

###Statement: {{statement}}

Tell me its output and corresponding reasoning steps in json format, with keywords \"valid\", \"reasoning_stesp\" and \"output\".
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

COMPARISON_INSTRUCTION = (
f"""You're a content writer, and your task is to write a statement the mian contents of the given ###Context and the comparison type two-hop question.

The input contains:
###Context: A title and some sentences.
###Question: A two-hop complex, and comparison question, in which only one-hop is relevant to the given ###Context.

###Explaination about comparison:
A comparison type question refers to a specific type of question that requires the comparison or contrast between two or more entities, concepts, or options. \
These questions typically ask for a judgment or evaluation based on the differences or similarities between the given options. For examples:
1. "Which city is larger, New York or Paris?"
2. "What are the differences between a comet and an asteroid?"
3. "Compare the advantages and disadvantages of solar energy and wind energy."
4. "Which is a more effective treatment for the common cold, medication A or medication B?"
In these examples, the questions explicitly ask for a comparison between two entities (cities, celestial objects, energy sources) or options (treatments). \
The answers to these questions require the identification and evaluation of the distinguishing features, characteristics, or properties of the entities or options being compared.

You can finish this as follow steps:
1. Identify what the ###Question asks, and find the relevant information in the ###Context.
2. Identify the common contents that the ###Context contains and the ###Question asks.
3. Ingore or move the information that ###Question asks but ###Context does not mention.
4. According to the common contents, write a question-oriented statement of the ###Context.

Here are 2 examples:
##Example 1:
###Context: 
- Summer Magic
- Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, \
and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.

###Question: Which movie was produced first, Summer Magic or Hocus Pocus?

###Reasoning steps:
- Identify what the ###Question asks: The ###Question asks for a comparison between two movies produced times.
- Identify the common content between the ###Context and the ###Question: the movie "Summer Magic."
- Remove the information that the ###Question asks but the ###Context does not mention: "Hocus Pocus."
- Write a question-oriented statement based on the common content: "Summer Magic is a Walt Disney Productions film produced in 1963."

###Statement: Summer Magic is a Walt Disney Productions film produced in 1963.

##Example 2:
###Context: 
- John Braine
- John Gerard Braine (13 April 1922 \u2013 28 October 1986) was an English novelist.
- Braine is usually listed among the Angry Young Men, a loosely defined group of English writers who emerged on the literary scene in the 1950s.

###Question: Were Patrick McCabe and John Braine of the same nationality?

###Reasoning steps:
- Identify what the ###Question asks: The ###Question asks for a comparison of nationality between two individuals, Patrick McCabe and John Braine.
- Identify the common content between the ###Context and the ###Question: John Braine.
- Remove the information that the ###Question asks but the ###Context does not mention: Patrick McCabe.
- Write a question-oriented statement based on the common content: "John Gerard Braine was an English novelist."

###Statement: John Gerard Braine was an English novelist.

Here is the input information:
###Context: 
{{context}}

###Question: {{question}}

Tell me its output and corresponding reasoning steps in json format, with keywords \"valid\", \"reasoning_stesp\" and \"statement\".
\"valid\" is True if you can answer it, else False.
""" 
)

FULL_ANSWER_INSTRUCTION = (
f"""You're a writer expert, and your task is to write a statement of the given ###Context, the comparison type two-hop ###Question and the corresponding ###Answer.

The input contains:
###Question: A two-hop complex, and comparison question, in which only one-hop is relevant to the given ###Context.

###Answer: The corresponding answer

###Context1: A title and some sentences.

###Context2: A title and some sentences.

###Explaination about comparison:
A comparison type question refers to a specific type of question that requires the comparison or contrast between two or more entities, concepts, or options. \
These questions typically ask for a judgment or evaluation based on the differences or similarities between the given options. For examples:
1. "Which city is larger, New York or Paris?"
2. "What are the differences between a comet and an asteroid?"
3. "Compare the advantages and disadvantages of solar energy and wind energy."
4. "Which is a more effective treatment for the common cold, medication A or medication B?"
In these examples, the questions explicitly ask for a comparison between two entities (cities, celestial objects, energy sources) or options (treatments). \
The answers to these questions require the identification and evaluation of the distinguishing features, characteristics, or properties of the entities or options being compared.

You can finish this as follow steps:
1. First check the ###Answer is yes/no or some specific contents.
2. Identify the relevant content to the ###Question ask in ###Context1.
3. Identify the relevant content to the ###Question ask in ###Context2.
4. Combine the ###Answer and ###Question according to the relevant content


Here are 2 examples:
##Example1
###Question: Which movie was produced first, Summer Magic or Hocus Pocus?

###Answer: Summer Magic

###Context1: 
- Summer Magic
- Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, \
and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.

###Context2: 
- Hocus Pocus (1993 film)
- Hocus Pocus is a 1993 American comedy horror fantasy film directed by Kenny Ortega, starring Bette Midler, Kathy Najimy and Sarah Jessica Parker; written by Neil Cuthbert and Mick Garris, and based on a story by Garris and David Kirschner.

###Reasoning steps:
- ###Answer is a specific content: Summer Magic.
- Identify the relevant content in ###Context1: Summer Magic as a 1963 Walt Disney Productions film.
- Identify the relevant content in ###Context2: Hocus Pocus as a 1993 American comedy horror fantasy film.
- Combine the answer and question according to the relevant content: The movie Summer Magic was produced first, as it is a 1963 Walt Disney Productions film, while Hocus Pocus is a 1993 American comedy horror fantasy film.

###Statement:  The movie Summer Magic was produced first, as it is a 1963 Walt Disney Productions film, while Hocus Pocus is a 1993 American comedy horror fantasy film.


##Example2
###Question: Were Patrick McCabe and John Braine of the same nationality?

###Answer: no

###Context1:
- Patrick McCabe (novelist)
- Patrick McCabe (born 27 March 1955) is an Irish writer.

###Context2: 
- John Braine
- John Gerard Braine (13 April 1922 \u2013 28 October 1986) was an English novelist.
- Braine is usually listed among the Angry Young Men, a loosely defined group of English writers who emerged on the literary scene in the 1950s.

###Reasoning steps
- ###Answer is no, indicating that Patrick McCabe and John Braine were not of the same nationality.
- Identify the relevant content in ###Context1: Patrick McCabe is an Irish writer.
- Identify the relevant content in ###Context2: John Braine was an English novelist and part of the Angry Young Men, a group of English writers.
- Combine the answer and question according to the relevant content: Patrick McCabe and John Braine were not of the same nationality. Patrick McCabe is Irish, whereas John Braine was English.

###Statement: Patrick McCabe and John Braine were not of the same nationality. Patrick McCabe is Irish, whereas John Braine was English.


##Example3
###Question: Are Rennae Stubbs and Carly Gullickson both tennis players?

###Answer: yes

###Context1:
- Rennae Stubbs
- Rennae Stubbs (born 26 March 1971) is an Australian retired tennis player.

###Context2: 
- Carly Gullickson
- Carly Gullickson (born November 26, 1986) is a former American professional tennis player.

###Reasoning steps
- The answer is "yes," indicating that both Rennae Stubbs and Carly Gullickson are tennis players.
- Identify the relevant content in Context1: Rennae Stubbs is an Australian retired tennis player.
- Identify the relevant content in Context2: Carly Gullickson is a former American professional tennis player.
- Combine the answer and question according to the relevant content: Rennae Stubbs and Carly Gullickson are both tennis players.

###Statement: Rennae Stubbs and Carly Gullickson are both tennis players.

Here is the input information:
###Question: {{question}}

###Answer: {{answer}}

###Context1:
{{context1}}

###Context2:
{{context2}}

Tell me its output and corresponding reasoning steps in json format, with keywords \"valid\", \"reasoning_stesp\" and \"statement\".
\"valid\" is True if you can answer it, else False.
"""
)

INSTRUCTIONS = {
    "summary": SUMMARY_INSTRUCTION,
    "refine": REFINE_INSTRUCTION,
    "comparison":COMPARISON_INSTRUCTION,
    "full_answer":FULL_ANSWER_INSTRUCTION
}


if __name__ == "__main__":
    print(SUMMARY_INSTRUCTION)
