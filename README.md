# Document_Clinical_BERT

 * Binary Classification & Multi-Label Classification

    * Phenotyping of Clinical Notes with Improved Document Classification Models Using Contextualized Neural Language Models [[paper]](https://arxiv.org/pdf/1910.13664v2.pdf)

    * Model Structures

      * Doc_BERT_LSTM
      
         * Repeat the BERT model as many times as the number of sentences
        
         * Collecting the CLS_token extracted through the BERT

         * Long Short Term Memory

         * Classifier


      * Doc_BERT_MEAN

         * Repeat the BERT model as many times as the number of sentences
        
         * Collecting the CLS_token extracted through the BERT

         * Global Average Pooling

         * Classifier
         
    * Evaluation metrics used in this code
    
      * macro-averaged F1
      * micro-averaged F1
    
