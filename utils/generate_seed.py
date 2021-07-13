import random


if __name__ == '__main__':
    # weight_decay =
    triplet = input('input triplet: ')
    triplet = [float(e) for e in triplet.split('\t')]
    while True:
        # if random.random() < 0.2:
        #     descending = True
        # else:
        #     descending = False
        for i, e in enumerate(triplet):
            # triplet[i] = random.uniform(0.95, 1.0) * e if descending else random.uniform(1.0, 1.10) * e
            triplet[i] = random.uniform(1, 1.5) * e
        print('\t'.join(map(lambda e: str(round(e, 3)), triplet)))
        # input('Enter')
        triplet = input('input triplet: ')
        triplet = [float(e) for e in triplet.split('\t')]