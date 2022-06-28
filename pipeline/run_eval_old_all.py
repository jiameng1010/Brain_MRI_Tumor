from pipeline import pipeline
import numpy as np

num_examples = 20
num_examples_real = 0

eval_list = ['samseg4_tumor_SVF0.5.nii.gz',
             'samseg4_tumor_SVF0.72.nii.gz',
             'samseg4_tumor_SVF0.82.nii.gz',]
if __name__ == '__main__':
    eval_dir = '/media/mjia/Seagate Backup Plus Drive/Researches/Volume_Segmentation/synthesised7_old/'
    num_of_error = 0

    for file_name in eval_list:
        mean0 = np.asarray([0.0, 0.0, 0.0])
        mean1 = np.asarray([0.0, 0.0, 0.0])

        num_examples_real = 0
        for i in range(num_examples):
            print(i)
            try:

                pp = pipeline(eval_dir+str(i).zfill(5), eval_dir+str(i).zfill(5)+'/output', 1500, True, resume=True)

                dice0 = pp.run_eval(file_name, with_cortex=False)
                dice1 = pp.run_eval(file_name, with_cortex=True)
                mean0 += dice0
                mean1 += dice1
                num_examples_real += 1
            except:
                continue
        print('******************')
        print(file_name)
        print(mean0/num_examples_real)
        print(mean1/num_examples_real)