# Kaggle_Competition_Toxic_Comments_Classification
Classify the data from a huge size of Wikipedia comments which have been labeled by human raters for toxic behavior.

The types of toxicity are:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

I created a model in Matlab which predicts a probability of each type of toxicity for each comment.

The algorithm first converts the words into vector based on GloVe, this is done via a thousand iterations of embedding training.
The acquired vector with 250 dimensions is then used to train a model based on Long-Short Term Memory Network.
You may download the data from the official competition site if you registered for the competition.
