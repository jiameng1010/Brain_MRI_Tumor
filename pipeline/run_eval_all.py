from pipeline import pipeline
import numpy as np

num_examples = 20
num_examples_real = 0

eval_list =  ['samseg_tumor_DISP_method30.75.nii.gz',
              'samseg_TUMOR_DISP_method30.75.nii.gz',
              'samseg_tumor_DISP_method30.45.nii.gz',
              'samseg_TUMOR_DISP_method30.45.nii.gz',
              ]

'''samseg_tumor_DISP0.5.nii.gz',
'samseg_TUMOR_DISP0.5.nii.gz',
'samseg_tumor_Samseg0.0.nii.gz',
'samseg_TUMOR_Samseg0.0.nii.gz',
'samseg_tumor_SVF_method10.05.nii.gz',
'samseg_TUMOR_SVF_method10.05.nii.gz',
'samseg_tumor_SVF_method10.25.nii.gz',
'samseg_TUMOR_SVF_method10.25.nii.gz',
'samseg_tumor_SVF_method20.05.nii.gz',
'samseg_TUMOR_SVF_method20.05.nii.gz',
'samseg_tumor_SVF_method20.25.nii.gz',
'samseg_TUMOR_SVF_method20.25.nii.gz',
]'''
if __name__ == '__main__':
    eval_dir = '/home/mjia/Researches/Volume_Segmentation/TumorMRI/TumorMRI_code/pipeline/test_data/synthesised7/'
    num_of_error = 0

    for file_name in eval_list:
        mean0 = np.asarray([0.0, 0.0, 0.0])
        mean1 = np.asarray([0.0, 0.0, 0.0])

        num_examples_real = 0
        for i in range(num_examples):
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