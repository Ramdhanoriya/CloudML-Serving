alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

dict = {}

for i, c in enumerate(alphabet):
    dict[c] = i + 1

key_list = dict.keys()

with open('v.txt', 'w') as file:
    for key in key_list:
        file.write(key+"\n")


