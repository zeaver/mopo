__all__ = ["INSTRUCTIONS"]

# demo input
#####################################################################################################################
# ```
# You're a content writer, and your task is to summarize the mian contents of the given input's supporting fact sentences and the given two-hop question.
 
# You can finish this as follow steps:
# 1. decompose the given two-hop question and generate a sub-question that is relevant to the given supporting fact sentences.
# 3. according to the given supporting fact sentences, covert the sub-questions into a declarative sentence.

# Here are 2 examples:

# ##Example 1:
# ###Question: Which movie was produced first, Summer Magic or Hocus Pocus?
# ###Type: Comparison

# ###Supporting Fact Sentences: 
# - Summer Magic
# - Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, \
# and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.

# ###Subquestion: When was the moive \"Summer Magic\" produced?

# ###Statement: Summer Magic is a Walt Disney Productions film produced in 1963.

# ###Explanation: The question is asking about the production order of two movies, \"Summer Magic\" and \Hocus Pocus\". \
# The supporting fact sentences provide information about "Summer Magic." \
# By decomposing the question, we generate the sub-question,\ "When was the movie \'Summer Magic\' produced?\" \
# According to the supporting fact sentences, the movie "Summer Magic" is a 1963 Walt Disney Productions film. \
# Therefore, the declarative statement is, "Summer Magic is a Walt Disney Productions film produced in 1963."

# ##Example 2:
# ###Question: What award was received in 1982 by the professor that supervised Jun Li's Harvard Ph.D?
# ###Type: Bridge

# ###Supporting Fact Sentences: 
# - Jun Li
# - He received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau.

# ###Subquestion: Who was the PhD supervisor of Jun Li in Harvard?

# ###Statement: Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University.

# ###Explanation: The question is asking about the award received by the professor who supervised Jun Li's Harvard Ph.D. \
# The supporting fact sentences provide information about Jun Li and his Ph.D. \
# By decomposing the question, we generate the sub-question, \"Who was the PhD supervisor of Jun Li in Harvard?\" \
# According to the supporting fact sentences, Jun Li received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau. \
# Therefore, the declarative statement is, "Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University."

# Here is the input information:
# ###Question: The 2011 UTSA Roadrunners football team were coached by the former head coach of what Florida college?
# ###Type: Bridge

# ###Supporting Fact Sentences: 
# - 2011 UTSA Roadrunners football team
# - The team was coached by veteran head football coach Larry Coker.

# Tell me its sub-question and corresponding statement in json format.
# Only use the information that's provided in the text.
# ```
#####################################################################################################################

INSTRUCTION = (
f"""You're a content writer, and your task is to summarize the mian contents of the given input's supporting fact sentences and the given two-hop question.
 
You can finish this as follow steps:
1. decompose the given two-hop question and generate a sub-question that is relevant to the given supporting fact sentences.
2. according to the given supporting fact sentences, covert the sub-questions into a declarative sentence.

Here are 2 examples:

##Example 1:
###Question: Which movie was produced first, Summer Magic or Hocus Pocus?
###Type: Comparison

###Supporting Fact Sentences: 
- Summer Magic
- Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, \
and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.

###Subquestion: When was the moive \"Summer Magic\" produced?

###Statement: Summer Magic is a Walt Disney Productions film produced in 1963.

###Explanation: The question is asking about the production order of two movies, \"Summer Magic\" and \Hocus Pocus\". \
The supporting fact sentences provide information about "Summer Magic." \
By decomposing the question, we generate the sub-question,\ "When was the movie \'Summer Magic\' produced?\" \
According to the supporting fact sentences, the movie "Summer Magic" is a 1963 Walt Disney Productions film. \
Therefore, the declarative statement is, "Summer Magic is a Walt Disney Productions film produced in 1963."

##Example 2:
###Question: What award was received in 1982 by the professor that supervised Jun Li's Harvard Ph.D?
###Type: Bridge
###BridgeInfo: Shing-Tung Yau

###Supporting Fact Sentences: 
- Jun Li
- He received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau.

###Subquestion: Who was the PhD supervisor of Jun Li in Harvard?

###Statement: Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University.

###Explanation: The question is asking about the award received by the professor who supervised Jun Li's Harvard Ph.D. \
The supporting fact sentences provide information about Jun Li and his Ph.D. \
THe ###BridgeInfo is \"Shing-Tung Yau\", so the sub-question is oriented/relevant to \"Shing-Tung Yau\".\
By decomposing the question, we generate the sub-question, \"Who was the PhD supervisor of Jun Li in Harvard?\" \
According to the supporting fact sentences, Jun Li received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau. \
Therefore, the declarative statement is, "Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University."\
And it is a sub-question-aware summary of the supporting fact sentences.

Here is the input information:
###Question: {{question}}
###Type: {{question_type}}

###Supporting Fact Sentences: 
{{supporting_facts}}

Only use the information that's provided in the text.
Tips: 
1. if the question type is **Bridge**, the sub_question is often relevant to the given ###BridgeInfo.
2. The statement is a sub_question-aware summary of the given ###Supporting Fact Sentences.
Tell me its sub_question and corresponding statement in json format, with keywords \"sub_question\" and \"statement\".
"""
)

# INSTRUCTIONS = (
# f"""You're a content writer, and your task is to summarize the mian contents of the given input's supporting fact sentences and the given question.
 
# You can finish this as follow steps:
# 1. decompose the given question and generate a sub-question
# 2. the sub-question is relevant to the given supporting fact sentences
# 3. according to the given supporting fact sentences, covert the sub-questions into a declarative sentence.

# Here are 2 examples:

# ##Example 1:
# ###Question: Which movie was produced first, Summer Magic or Hocus Pocus?

# ###Supporting Fact Sentences: 
# - Summer Magic
# - Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, \
# and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.
# - The film was based on the novel \"Mother Carey's Chickens\" by Kate Douglas Wiggin and was directed by James Neilson.

# ###Subquestion: When was the moive \"Summer Magic\" produced?

# ###Statement: Summer Magic is a Walt Disney Productions film produced in 1963.

# ###Explanation: The question is asking about the production order of two movies, \"Summer Magic\" and \Hocus Pocus\". \
# The supporting fact sentences provide information about "Summer Magic." \
# By decomposing the question, we generate the sub-question,\ "When was the movie \'Summer Magic\' produced?\" \
# According to the supporting fact sentences, the movie "Summer Magic" is a 1963 Walt Disney Productions film. \
# Therefore, the declarative statement is, "Summer Magic is a Walt Disney Productions film produced in 1963."

# ##Example 2:
# ###Question: What award was received in 1982 by the professor that supervised Jun Li's Harvard Ph.D?

# ###Supporting Fact Sentences: 
# - Jun Li
# - Jun Li is a Chinese mathematician who is currently a Professor of Mathematics at Stanford University.
# - He focuses primarily on moduli problems in algebraic geometry and their applications to mathematical physics, geometry and topology.
# - He received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau.

# ###Subquestion: Who was the PhD supervisor of Jun Li in Harvard?

# ###Statement: Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University.

# ###Explanation: The question is asking about the award received by the professor who supervised Jun Li's Harvard Ph.D. \
# The supporting fact sentences provide information about Jun Li and his Ph.D. \
# By decomposing the question, we generate the sub-question, \"Who was the PhD supervisor of Jun Li in Harvard?\" \
# According to the supporting fact sentences, Jun Li received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau. \
# Therefore, the declarative statement is, "Shing-Tung Yau was the PhD supervisor of Jun Li at Harvard University."

# Here is the input information:
# ###Question: The 2011 UTSA Roadrunners football team were coached by the former head coach of what Florida college?

# ###Supporting Fact Sentences: 
# - 2011 UTSA Roadrunners football team
# - The 2011 UTSA Roadrunners football team represented the University of Texas at San Antonio in the 2011 NCAA Division I FCS football season.
# - It was the first year of play for UTSA.
# - The team was coached by veteran head football coach Larry Coker.

# Tell me its sub-question and corresponding statement in json format.
# Only use the information that's provided in the text.

# """
# )

# INSTRUCTIONS = (
# f"""You're a content writer, and your task is to summarize the mian contents of the given input's supporting fact sentences and the given question. \
# Only use the information that's provided in the text. 

# You need to write a one-sentence statement that specifically follows:
# - 1. relavant to the given question.
# - 2. summary of the given supporting fact sentences.
# - 3. Do not appears the contents that supporting fact did not mention. 
# - 4. Follow Supporting Fact Sentences expression and style as much as possible

# Here are 2 examples:

# ###Example 1:
# ###Question: Which movie was produced first, Summer Magic or Hocus Pocus?

# ###Supporting Fact Sentences: 
# - Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, \
# and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.
# - The film was based on the novel \"Mother Carey's Chickens\" by Kate Douglas Wiggin and was directed by James Neilson.

# ###Subquestion: Summer Magic is a Walt Disney Productions film produced in 1963.

# ###Explanation: The statement "Summer Magic is a Walt Disney Productions film produced in 1963" follows the given question about the production order of Summer Magic and Hocus Pocus. \
# The statement summarizes the supporting fact sentences, which mention that Summer Magic is a film produced by Walt Disney Productions in 1963.

# ###Example 2:
# ###Question: What award was received in 1982 by the professor that supervised Jun Li's Harvard Ph.D?

# ###Supporting Fact Sentences: 
# - Jun Li () is a Chinese mathematician who is currently a Professor of Mathematics at Stanford University.
# - He focuses primarily on moduli problems in algebraic geometry and their applications to mathematical physics, geometry and topology.
# - He received his Ph.D. from Harvard University in 1989, under the supervision of Shing-Tung Yau.

# ###Statement: Summer Magic is a Walt Disney Productions film produced in 1963.

# """
# )

if __name__ == "__main__":
    print(INSTRUCTION)
