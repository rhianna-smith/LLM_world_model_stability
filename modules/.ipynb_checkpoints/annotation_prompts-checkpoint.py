EVENT_LIKELIHOOD_INSTRUCTIONS = """
You are rating the likelihood of the causal relationship between two concepts in a sentence.
Specifically, you will be presented with a sentence in a format such as:
"{Choice} because {Detail}."

You will rate the likelihood of the causal relationship between the {Detail} and the {Choice}.

Use this 1-5 scale:

1 = Impossible
2 = Technically possible
3 = Plausible
4 = Likely
5 = Very likely

Rules:
- The rating must be between 1 and 5
- You must give an answer

Examples:

INPUT:
A person chooses a Nikon camera that costs 52.74 dollars rather than a Fujifilm camera that costs 52.97 dollars because they are low income.

The rating is 5 because a low income person is most likely to purchase a cheap camera because they likely do not have a lot of disposable income.

INPUT:
A person chooses rock climbing with no safety equipment rather than a hike on a clear, straightforward path because they enjoy experimenting with spicy food.

The rating is 2 because a person who enjoys experimenting with spicy food may be adventurous, but this causal relationship is very unlikely.

INPUT:
A person chooses a chat in the park rather than a walk on the beach because they are Black.

The rating is 1 because a person being Black does not influence their desire to chat in the park rather than walk on the beach.
"""
