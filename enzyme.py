import pandas as pd
import numpy as np
import os
import multiprocessing as mp

parent_dir = os.path.abspath(os.path.dirname(__file__))


df = pd.read_csv(parent_dir+"/dataset/train/cleaned_data_bi.csv",sep='\t',header=0)
train_sequences = []
for seq in train_sequences:
    if 2<len(seq)<5:
        train_sequences.append(seq)


def generate_all_tripeptide_sequences():
    amino_acids = ['G', 'A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y', 'S', 'T', 'C', 'P', 'N', 'Q', 'H', 'K', 'R', 'E', 'D']
    tripeptide_sequences = []
    for amino_acid1 in amino_acids:
        for amino_acid2 in amino_acids:
            for amino_acid3 in amino_acids:
                tripeptide_sequence = amino_acid1 + amino_acid2 + amino_acid3
                tripeptide_sequences.append(tripeptide_sequence)
    return tripeptide_sequences

def generate_all_tetrapeptide_sequences():
    amino_acids = ['G', 'A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y', 'S', 'T', 'C', 'P', 'N', 'Q', 'H', 'K', 'R', 'E', 'D']
    tetrapeptide_sequences = []
    for amino_acid1 in amino_acids:
        for amino_acid2 in amino_acids:
            for amino_acid3 in amino_acids:
                for amino_acid4 in amino_acids:
                    tetrapeptide_sequence = amino_acid1 + amino_acid2 + amino_acid3 + amino_acid4
                    tetrapeptide_sequences.append(tetrapeptide_sequence)
    return tetrapeptide_sequences


def single_enzyme(food):
    input_file = parent_dir+"/dataset/rpg/proteome/"+food
    out_file = parent_dir+f"/dataset/rpg/proteome_enzymolysis/{food}.csv"
    os.system(f"rpg -i {input_file} -e 33 42 -d c -o {out_file}")

def enzyme():
    foods = os.listdir(parent_dir+"/dataset/rpg/proteome")
    print(foods)
    pool = mp.Pool(len(foods)) 
    results = [pool.apply_async(func=single_enzyme, args=(food,)) for food in foods]        
    results = [p.get() for p in results]
    print('结束测试')


def single_process(food):
    with open(parent_dir+f"/dataset/rpg/ep_screen/{food}.csv",'w') as fw:
        ls = []
        in_file = parent_dir+f"/dataset/rpg/proteome_enzymolysis/{food}.csv"
        with open(in_file, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                if not line.startswith('>') and (20 > len(line.strip()) > 2) :
                    ls.append(line.strip())
        ls = list(set(ls))
        for seq in ls:
            if 2<len(seq)<5 and seq not in train_sequences:
                fw.write(seq+"\t"+food+"\n")
        print(food,': run ok')

def together():
    foods = os.listdir(parent_dir+"/dataset/rpg/proteome")
    pool = mp.Pool(20) 
    results = [pool.apply_async(func=single_process, args=(food,)) for food in foods]        
    results = [p.get() for p in results]
    print('ok')

def statistics():
    with open(parent_dir+f"/dataset/rpg/statistics.csv",'w') as f:
        f.write('food\tnumber_of_ep\tnumber_of_ep_screen\n')
        foods = os.listdir(parent_dir+"/dataset/rpg/proteome")
        totoal_num_ep = 0
        total_num_ep_screen = 0
        for food in foods:
            ep = parent_dir+f"/dataset/rpg/proteome_enzymolysis/{food}.csv"
            num_ep = int(len(open(ep, 'r').readlines())/2)
            ep_screen = parent_dir+f"/dataset/rpg/ep_screen/{food}.csv"
            num_ep_screen = len(open(ep_screen, 'r').readlines())
            totoal_num_ep += num_ep
            total_num_ep_screen += num_ep_screen
            f.write(food+'\t'+str(num_ep)+'\t'+str(num_ep_screen)+'\n')
        f.write('total\t'+str(totoal_num_ep)+'\t'+str(total_num_ep_screen)+'\n')

def generate_34ep():
    with open(parent_dir+f"/dataset/rpg/34_ep.csv",'w') as fw:
        fw.write('Sequence\n')
        foods = os.listdir(parent_dir+"/dataset/rpg/proteome")
        ls = []
        for food in foods:
            ep_screen = parent_dir+f"/dataset/rpg/ep_screen/{food}.csv"
            with open(ep_screen, 'r') as fr:
                lines = fr.readlines()
                for line in lines:
                    if not 'X' in line:
                        ls.append(line.split('\t')[0])
        ls = list(set(ls))
        for seq in ls:
            fw.write(seq+"\n")


if __name__ == '__main__':
    # enzyme()
    # together()
    # statistics()
    generate_34ep()
    pass