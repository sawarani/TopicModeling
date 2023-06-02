# Topic Modeling of popular science articles

This is an overview of several TM techniques including:
* [LDA](https://radimrehurek.com/gensim/models/ldamodel.html "Latent Dirichlet Allocation")
* [ATM](https://radimrehurek.com/gensim/models/atmodel.html "Author-Topic Model")
* [GLDA](https://guidedlda.readthedocs.io/en/latest/ "Guided Latent Dirichlet Allocation")
* [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html "Non-negative Matrix Factorization")
* [CTM](https://contextualized-topic-models.readthedocs.io/en/latest/introduction.html "Contextualized Topic Model")
* [BERTopic](https://maartengr.github.io/BERTopic/ "BERTopic")
* [BTM](https://bitermplus.readthedocs.io/en/latest/ "Biterm Topic Model")

The corpus used is a compilation of Russian popular science texts samped from Elementy bolshoi nauki (https://elementy.ru/), an online media outlet covering various aspects of natural sciences and technology. It contains 2,289 popular science articles published between 2010 and 2023, or approximately 3m words. Preprocessing techniques include tokenization, lemmatization with pymorphy2, named entity recognition with spaCy, and collocation extraction with Gensim module Phrases.
