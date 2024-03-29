===========
Socio Probe
===========

Code has been built on two simple abstractions:

The first is **Embedder** that is used to create embeddings extracted from specific layers and save them to disk.

The second in is the **Probers** that train classifiers for porbing.

A complete run example can be found in **run_p.py**.

Example
-------

.. code-block:: python


        embe = Embedder(m)
        embe.create_embeddings(dataset["text"].tolist(),
                               dataset["label"].tolist(),
                               list(range(1, embe.model.config.num_hidden_layers+1)),
                               f"embeddings/embs_save.pkl")

        prober = ClassicalProber(embe.model.config.hidden_size)
        
        macro_f_dict = prober.run(f"embeddings/embs_save.pkl")
        for layer in macro_f_dict.items():

            f1 = layer[1]['f1']
            loss = layer[1]['loss']


Refs
----

We use early stopping code coming from: https://github.com/Bjarten/early-stopping-pytorch
