import data_helpers

maxlen = 9


vocab, _, vocab_size, check = data_helpers.create_vocab_set()

x_test = ['this is awesome']

res = data_helpers.encode_data(x_test, maxlen, vocab, 69, check)

print(res)