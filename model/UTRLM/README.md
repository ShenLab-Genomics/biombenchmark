---
title: UTR-LM

app_file: app.py

python_version: 3.9.4

pinned: true

license: bsd

sdk: streamlit

---

# UTR-LM: A Semi-supervised 5’ UTR Language Model for mRNA Translation and Expression Prediction

The untranslated region (UTR) of an RNA molecule plays a vital role in gene expression regulation. Specifically, the 5' UTR, located at the 5' end of an RNA molecule, is a critical determinant of the RNA’s translation efficiency. Language models have demonstrated their utility in predicting and optimizing the function of protein encoding sequences and genome sequences. In this study, we developed a semi-supervised language model for 5’ UTR, which is pre-trained on a combined library of random 5' UTRs and endogenous 5' UTRs from multiple species. We augmented the model with supervised information that can be computed given the sequence, including the secondary structure and minimum free energy, which improved the semantic representation. 

# Data
All datasets are saved in the "data" folder.

# Source Code
The Source Code and detailed README.md could find in [link](https://github.com/a96123155/UTR-LM/).

# Demo
To run this Demo, you could directly click the "Run with parameters" in this page. You could read the file [run] to modify the parameter.


# Prediction of MRL
The Webserver of UTR-LM is available at [link](https://huggingface.co/spaces/yanyichu/UTR_LM).
On the App page, enter the fasta format of the 5'UTR sequence, and then click the "Predict" button. Sometimes after a short wait, the results will appear later on the page and you can download.