import os
from tqdm import tqdm

def extract_hmm(fasta_dir, hmm_dir, hmm_db_path, cpu=20):
    filename_list = os.listdir(fasta_dir)
    for filename in tqdm(filename_list):
        fasta_path = os.path.join(fasta_dir, filename)
        hmm_path = os.path.join(hmm_dir, filename.replace('fa', 'txt'))
        cmd = 'hhblits -i %s -o /dev/null -ohhm  %s -d %s -cpu %s  > shell.out 2>&1' % (fasta_path, hmm_path, hmm_db_path, cpu)
        os.system(cmd)

if __name__ == '__main__':
    fasta_dir = '/staff/wangzhaohui/proteinFold/data/fasta'
    hmm_dir = '/staff/wangzhaohui/proteinFold/data/hmm_uniclust30_2018_08'
    hmm_db_path = '/staff/wangzhaohui/proteinFold/database/uniclust30/uniclust30_2018_08/uniclust30_2018_08'
    cpu = 60
    extract_hmm(fasta_dir, hmm_dir, hmm_db_path, cpu)