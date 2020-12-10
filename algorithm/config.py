
import os
from pathlib import Path


HOME_PATH = str(Path.home())
DATA_IMG_EXTENSION_PATH = os.path.join(HOME_PATH, 'JupyterLab/data/ai_stomach/img_data')
DATA_LABEL_EXTENSION_PATH = os.path.join(HOME_PATH, 'JupyterLab/data/ai_stomach/label_data')
TRG_LABEL_PATH = os.path.join(HOME_PATH, 'JupyterLab/data/ai_stomach/nii_status/all_status_0918.xlsx')
SURVIVAL_LABEL_PATH = os.path.join(HOME_PATH, 'JupyterLab/data/ai_stomach/survival_data.xlsx')

ROI_EXPANSION = 10
N_DILATION = 5

ERROR_IMAGE_OPEN = 'Unable to decode CT image'
ERROR_DIM_MISMATCH = 'Dimension of image and label mismatch'
ERROR_LESION_COUNT = 'Lesion count mismatch'
ERROR_LESION_RELATIONSHIP = 'Lesion id-layers relationship mismatch'
ERROR_LABEL_IN_TWO_LAYER = 'Same label occur in more than one layer'
