# Downloads example corpora and vectors for structural probing.
# Includes conllx files, raw text files, and ELMo contextual word representations

# By default, downloads a (very) small subset of the EN-EWT
# universal dependencies corpus. 

# For demo purposes, also downloads pre-trained probes on BERT-large.

wget https://nlp.stanford.edu/~johnhew/public/en_ewt-ud-sample.tgz
wget https://nlp.stanford.edu/~johnhew/public/sp/bertlarge16-distance-probe.params
wget https://nlp.stanford.edu/~johnhew/public/sp/bertlarge16-depth-probe.params
tar xzvf en_ewt-ud-sample.tgz
mkdir -p example/data
mv en_ewt-ud-sample example/data
mv bertlarge16-distance-probe.params bertlarge16-depth-probe.params example/data
rm en_ewt-ud-sample.tgz
