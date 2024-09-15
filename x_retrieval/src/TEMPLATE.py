SUMMARY_INSTRUCTION = """You're a content writer, and your task is to partly summarize the mian contents of the given ###Context and the two-hop question.

The input contains:
###Context: A title and some sentences.
###Question: A two-hop complex question, in which only one-hop is relevant to the given ###Context.
Question contains two types, include \"bridge\" and \"comparison\". 
- Bridge question is often a clause, we need to found the answer to the main or subordinate part of the ###Question.
- A comparison type question refers to a specific type of question that requires the comparison or contrast between two or more entities, concepts, or options.
 
You can finish this as follow steps:
1. Find the common contents that the ###Context contains and the ###Question asks.
2. According to the common contents, write a question-oriented summary of the ###Context.
3. Remove the information that is not appears in ###Context.

Here are 2 examples:

##Example 1:
###Context: 
- Summer Magic
- Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine. The film was based on the novel \"Mother Carey's Chickens\" by Kate Douglas Wiggin and was directed by James Neilson. This was the fourth of six film Mills did for Disney, and the young actress received a Golden Globe nomination for her work here.

###Question: Which movie was produced first, Summer Magic or Hocus Pocus?

###Reasoning steps: 
1. The common context between the ###Context and the two-hop ###Question is the movie "Summer Magic." 
2. The ###Context provides information about the movie, stating that it is a 1963 Walt Disney Productions film. 
Therefore, we can summarizes this information: "Summer Magic is a Walt Disney Productions film produced in 1963."

###Abstract: Summer Magic is a Walt Disney Productions film produced in 1963.

##Example 2:
###Question: What award was received in 1982 by the professor that supervised Jun Li's Harvard Ph.D?

###Context: 
- Jun Li
- Jun Li is a Chinese mathematician who is currently a Professor of Mathematics at Stanford University. He focuses primarily on moduli problems in algebraic geometry and their applications to mathematical physics, geometry and topology. He received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau.

###Reasoning steps: 
1. The common context between the ###Context and the two-hop ###Question is the person "Jun Li" and his Ph.D. at Harvard University.
2. The ###Context provides information about Jun Li, stating that he received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau. 
Therefore, we abstract this information by: "Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University."

###Abstract: Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University.
"""

SUMMARY_TEMPLATE = (
f"""
Here is the input information:
###Context: 
{{context}}

###Question: {{question}}

Only use the information that's provided in the text.

Generate the ###Abstract:
"""
)

FLAN_T5_INSTRUCTION = (
f"""Generate a query-focused summary.
Query:
{{question}}

Context:
{{context}}
"""
)