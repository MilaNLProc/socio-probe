==============
Social Probing
==============

Example
-------

.. code-block:: python

    from probers import MLDProber
    from sentence_transformers import SentenceTransformer
    import pandas as pd

    st = SentenceTransformer("nyu-mll/roberta-med-small-1M-3")

    mldprober = MLDProber(st, 512)

    english_train = "https://github.com/MilaNLProc/translation_bias/raw/master/data/en_us/en_us_TRAIN.xlsx"
    english_test = "https://github.com/MilaNLProc/translation_bias/raw/master/data/en_us/en_us_TEST.xlsx"


    english_train = pd.read_excel(english_train)
    english_train = english_train.dropna()


    mldprober.run(english_train["text"].values.tolist(), english_train["gender"].values.tolist())
