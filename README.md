# disastAIr
*Evaluating Relative Efficiency of ML Models for Classification of Disaster Tweets using NLP*

**disastAIr** is a student project as part of the [Machine Learning course](https://mlvu.github.io/) at Vrije Universiteit Amsterdam. The project encompasses training and evaluation of two different ML models for classification of disaster related Tweets using NLP.

While the project is based on a [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started), the aim of the project goes beyond that of the competition. Namely, the project sets a focus on evaluating different ML models not only based on their performance but also on their footprint cost. 

The models used are:
- a classical model in the form of a Markov classifier, and
- a deep learning (DL) model in the form of a distilled version of the transformer-based BERT model ([DistilBERT](https://arxiv.org/abs/1910.01108)).

Python is used for all models and experiments. The DL model is used through the `KerasNLP` library, while the Markov classifier is implemented fully from scratch.

A full project description, including results and a discussion, is given in the project report.

## Authors
| Name | Profile |
|---|---|
| Maria P. Jimenez M. | [GitHub](https://github.com/mapa21), [LinkedIn](https://www.linkedin.com/in/maria-paula-jim%C3%A9nez-moreno-59593221b/) |
| Lennart K.M. Schulz | [GitHub](https://github.com/lkm-schulz), [LinkedIn](https://www.linkedin.com/in/lkm-schulz/) |
| Laura I.M. Stampf | [GitHub](https://github.com/laustam), [LinkedIn](https://www.linkedin.com/in/laura-stampf/) |
| Dovydas Vadišius | [GitHub](https://github.com/DovydasVad), [LinkedIn](https://www.linkedin.com/in/dovydas-vadisius/) |