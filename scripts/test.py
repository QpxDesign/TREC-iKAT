with open("data/passage-index-splade/small.jsonl", "r") as f:
    for line in f.readlines():
        l = line.replace('{"id": ', '')
        l = line.split(',')[0]
        l = l.split(' ')[1]
        print(len(l))
