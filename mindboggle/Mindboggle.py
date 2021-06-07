import os
import subprocess
import mindboggle.util
class Mindboggle:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_path_brains = os.path.join(self.data_path, 'mindboggle_manually_labeled_individual_brains')
        self.data_path_atlases = os.path.join(self.data_path, 'mindboggle_atlases')
        self.data_path_templates = os.path.join(self.data_path, 'mindboggle_templates')
        self.subject_list = mindboggle.util.load_subject_list(self.data_path_brains)

    def get_subject_list(self):
        return self.subject_list

    def unzip_volume_file(self, group_name):
        zip_file_name = self.data_path_brains + '/' + group_name + '_volumes.tar.gz'
        subprocess.run(['tar', 'xvzf', zip_file_name, '-C', self.data_path_brains])

    def unzip_surface_file(self, group_name):
        zip_file_name = self.data_path_brains + '/SurfaceLabels_' + group_name + '.tar.gz'
        subprocess.run(['tar', 'xvzf', zip_file_name, '-C', self.data_path_brains])

    def get_atlas(self, index=0):
        atlas_files = ['OASIS-TRT-20_jointfusion_DKT31_CMA_label_probabilities_in_MNI152_v2.nii.gz',
                       'OASIS-TRT-20_jointfusion_DKT31_CMA_label_probabilities_in_OASIS-30_v2.nii.gz',
                       'OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_2mm_v2.nii.gz',
                       'OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz',
                       'OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_OASIS-30_v2.nii.gz']
        the_file = os.path.join(self.data_path_atlases, atlas_files[index])
        return mindboggle.util.load_gz_file(the_file)
