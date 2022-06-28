from pipeline import pipeline
import numpy as np

mean1 = np.asarray([0.0, 0.0, 0.0])
mean2 = np.asarray([0.0, 0.0, 0.0])
mean3 = np.asarray([0.0, 0.0, 0.0])
num_examples = 40
num_examples_real = 0
if __name__ == '__main__':
    org_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised5/'
    eval_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/'
    num_of_error = 0

    for i in range(num_examples):
        print(i)
        try:

            pp = pipeline(eval_dir+str(i).zfill(5), eval_dir+str(i).zfill(5)+'/output', 1500, True, resume=True)
            dice3 = pp.run_eval('samseg_tumor_SVF_method20.25.nii.gz', with_cortex=False)
            dice1 = pp.run_eval('samseg_TUMOR_DISP0.0.nii.gz', with_cortex=False)
            print(dice1)
            #pp_org = pipeline(eval_dir+str(i).zfill(5), org_dir+str(i).zfill(5)+'/output', 1500, True, resume=True)
            dice2 = pp.run_eval('samseg_tumor_SVF_method10.25.nii.gz', with_cortex=False)
            print(dice2)
            print(dice2 - dice1)

            print(dice3)
            print(dice3 - dice1)
            mean1 += dice1
            mean2 += dice2
            mean3 += dice3
            num_examples_real += 1
        except:
            continue
        print('--------------')


print('******************')
print(mean1/num_examples_real)
print(mean2/num_examples_real)
print(mean3/num_examples_real)