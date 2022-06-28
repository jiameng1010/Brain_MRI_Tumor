from pipeline import pipeline
import numpy as np

mean1 = np.asarray([0.0, 0.0, 0.0])
mean2 = np.asarray([0.0, 0.0, 0.0])
num_examples = 20
if __name__ == '__main__':
    eval_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised5/'
    output_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised5/plot/'
    num_of_error = 1
    for i in range(12, num_examples):
        pp = pipeline(eval_dir+str(i).zfill(5), eval_dir+str(i).zfill(5)+'/output', 1500, True, resume=True)
        dice1 = pp.plot_result('samseg1.nii.gz', 'samseg3_010.nii.gz', output_dir+str(i).zfill(5)+'.jpg')

print('******************')
print(mean1/num_examples)
print(mean2/num_examples)
