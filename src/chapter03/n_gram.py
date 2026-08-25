# n_gram.py

import collections

# N = 2, no padding.
# P(datawhale agent learns)

# corpus -> tokens
corpus1 = "datawhale agent learns datawhale agent works"
tokens1 = corpus1.split()
total_tokens = len(tokens1)

# --- 1. calculate P(datawhale) ---
count_datawhale = tokens1.count('datawhale')
p_datawhale = count_datawhale / total_tokens
print(f"1. P(datawhale) = {count_datawhale}/{total_tokens} = {p_datawhale:.3f}")

# --- 2. calculate P(agent|datawhale) ---
# get bigrams for the next.
bigrams = zip(tokens1, tokens1[1:])
bigram_counts = collections.Counter(bigrams)
count_datawhale_agent = bigram_counts[('datawhale', 'agent')]

p_agent_given_datawhale = count_datawhale_agent / count_datawhale
print(f"2. P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = "
      f"{p_agent_given_datawhale:.3f}")

# --- 3. calculate P(learns|agent) ---
count_agent_learns = bigram_counts[('agent', 'learns')]
count_agent = tokens1.count('agent')
p_learns_given_agent = count_agent_learns / count_agent
print(f"3. P(learns|agent) = {count_agent_learns}/{count_agent} = "
      f"{p_learns_given_agent:.3f}")

# --- 4. multiply all above to get P(datawhale agent learns) ---
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(f"4. P('datawhale agent learns') ≈ {p_datawhale:.3f} * "
      f"{p_agent_given_datawhale:.3f} * {p_learns_given_agent:.3f} = "
      f"{p_sentence:.3f}\n")


# N = 3, with padding.
# P(I am happy)

# corpus -> tokens
corpus2 = ["I am happy", "I am sad", "she is happy"]
tokens2 = []
total_tokens = 0
for corpus in corpus2:
      tokens = corpus.split()
      # insert padding.
      tokens.insert(0, "<pad>")
      tokens.insert(0, "<pad>")
      tokens.append("<pad>")

      total_tokens += len(tokens)
      tokens2.append(tokens)

# get bigrams.
bigram_counts = dict()
for tokens in tokens2:
      bigrams = zip(tokens, tokens[1:])
      bigram_count = collections.Counter(bigrams)
      for k, v in bigram_count.items():
            seq_count = bigram_counts.get(k, 0) + v
            bigram_counts[k] = seq_count

# print(bigram_counts)

# get trigrams.
trigram_counts = dict()
for tokens in tokens2:
      trigrams = zip(tokens, tokens[1:], tokens[2:])
      trigram_count = collections.Counter(trigrams)
      for k, v in trigram_count.items():
            seq_count = trigram_counts.get(k, 0) + v
            trigram_counts[k] = seq_count

# print(trigram_counts)

# --- 1. calculate P(I | <pad> <pad>) ---
count_pad_pad_I = trigram_counts[('<pad>', '<pad>', 'I')]
count_pad_pad = bigram_counts[('<pad>', '<pad>')]
p_pad_pad_I = count_pad_pad_I / count_pad_pad
print(f"1. P(I | <pad> <pad>) = {count_pad_pad_I}/{count_pad_pad} = {p_pad_pad_I:.3f}")

# --- 2. calculate P(am | <pad> I) ---
count_pad_I_am = trigram_counts[('<pad>', 'I', 'am')]
count_pad_I = bigram_counts[('<pad>', 'I')]
p_pad_I_am = count_pad_I_am / count_pad_I
print(f"2. P(am | <pad> I) = {count_pad_pad_I}/{count_pad_I} = {p_pad_I_am:.3f}")

# --- 3. calculate P(happy | I am) ---
count_I_am_happy = trigram_counts[('I', 'am', 'happy')]
count_I_am = bigram_counts[('I', 'am')]
p_I_am_happy = count_I_am_happy / count_I_am
print(f"3. P(happy | I am) = {count_I_am_happy}/{count_I_am} = {p_I_am_happy:.3f}")

# --- 4. calculate P(<pad> | am happy) ---
count_am_happy_pad = trigram_counts[('am', 'happy', '<pad>')]
count_am_happy = bigram_counts[('am', 'happy')]
p_am_happy_pad = count_am_happy_pad / count_am_happy
print(f"4. P(<pad> | am happy) = {count_am_happy_pad}/{count_am_happy} = {p_am_happy_pad:.3f}")

'''
 5. P('I am happy') = P('<pad> <pad> I am happy <pad>')
                    = P(I | <pad> <pad>) * P(am | <pad> I) * P(happy | I am) * P(<pad> | am happy)
'''

p_sentence = p_pad_pad_I * p_pad_I_am * p_I_am_happy * p_am_happy_pad
print(f"5. P('I am happy') ≈ {p_pad_pad_I:.3f} * "
      f"{p_pad_I_am:.3f} * {p_I_am_happy:.3f} * {p_am_happy_pad:.3f} = "
      f"{p_sentence:.3f}")