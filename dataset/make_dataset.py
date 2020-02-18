"""
CSE 517 - Final Project

This script outputs a file named sts.csv that contains the data to be used for analysis.
"""


if __name__ == '__main__':
    data = []
    all_domains = {
        '2012TRAIN': ['MSRpar', 'MSRvid', 'SMTeuroparl'],
        '2012GOLD': ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWn', 'surprise.SMTnews'],
        '2013GOLD': ['FNWN', 'headlines', 'OnWN'],
        '2014GOLD': ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news'],
        '2015GOLD': ['answers-forums', 'answers-students', 'belief', 'headlines', 'images']
    }
    for year, domains in all_domains.items():
        for domain in domains:
            with open(f'data/{year}/STS.gs.{domain}.txt', 'r') as f:
                gs = [line.strip() for line in f]
            with open(f'data/{year}/STS.input.{domain}.txt', 'r') as f:
                input = [line.strip() for line in f]
            for i, line in enumerate(input):
                sent1, sent2 = line.split('\t')
                data.append({
                    'Dataset': f'STS{year}',
                    'Domain': domain,
                    'Score': gs[i],
                    'Sent1': sent1,
                    'Sent2': sent2
                })

    with open('sts.csv', 'w') as fout:
        fout.write(f"Dataset\tDomain\tScore\tSent1\tSent2\n")
        for x in data:
            fout.write(f"{x['Dataset']}\t{x['Domain']}\t{x['Score']}\t{x['Sent1']}\t{x['Sent2']}\n")
