import msm
from utils.tokenization import Vocab


alphabet = msm.data.Alphabet.from_architecture("rna language")
vocab = Vocab.from_esm_alphabet(alphabet)
print(vocab.tokens)
