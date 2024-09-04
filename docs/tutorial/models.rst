Model Selection
===============

Here we will choose one of our QCML models and parameterize it for training. There are several QCML models that you can choose from and we are continuing to develop new models as well as improve existing ones. You can see which models are currently available on the sidebar.
about:blank#blocked
There are two categories of models. The first is a stand alone QCML model, which has at its core a single cost function that involves the quantum state. The second is a QCML PyTorch layer which has been built so that you can integrate it with classical ML deep learning layers, and so develop a hybrid quantum-classical model whose architecture fits your needs. The cost function is defined by you which is a classical cost function where QCML will be a part of the network.

Setting up each is a little different, so we'll cover them separately.

.. toctree::
   :caption: QCML Models

   /tutorial/stand_alone/index
   /tutorial/pytorch/index

