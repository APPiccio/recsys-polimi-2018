from tqdm import tqdm

train_sequential = open("train_sequential.csv", "r")
train = open("train.csv", "r")
output = open("new_train.csv", "w")
target = open("target_playlists.csv", "r")

target = list(target)[1:]
train = list(train)
output.write(train[0])
train = train[1:]
train_sequential = list(train_sequential)[1:]

target_list = list()
for line in target[:5000]:
    target_list.append(int(line))

j = 0
for line in tqdm(train):
    id = int(line.split(",")[0])
    if id in target_list:
        output.write(train_sequential[j])
        j += 1
    else:
        output.write(line)

output.close()


