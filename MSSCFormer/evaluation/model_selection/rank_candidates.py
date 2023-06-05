#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from MSSCFormer.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "MSSCFormerPlans"

    overwrite_plans = {
        'MSSCFormerTrainerV2_2': ["MSSCFormerPlans", "MSSCFormerPlansisoPatchesInVoxels"], # r
        'MSSCFormerTrainerV2': ["MSSCFormerPlansnonCT", "MSSCFormerPlansCT2", "MSSCFormerPlansallConv3x3",
                            "MSSCFormerPlansfixedisoPatchesInVoxels", "MSSCFormerPlanstargetSpacingForAnisoAxis",
                            "MSSCFormerPlanspoolBasedOnSpacing", "MSSCFormerPlansfixedisoPatchesInmm", "MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_warmup': ["MSSCFormerPlans", "MSSCFormerPlansv2.1", "MSSCFormerPlansv2.1_big", "MSSCFormerPlansv2.1_verybig"],
        'MSSCFormerTrainerV2_cycleAtEnd': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_cycleAtEnd2': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_reduceMomentumDuringTraining': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_graduallyTransitionFromCEToDice': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_independentScalePerAxis': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_Mish': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_Ranger_lr3en4': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_fp32': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_GN': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_momentum098': ["MSSCFormerPlans", "MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_momentum09': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_DP': ["MSSCFormerPlansv2.1_verybig"],
        'MSSCFormerTrainerV2_DDP': ["MSSCFormerPlansv2.1_verybig"],
        'MSSCFormerTrainerV2_FRN': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_resample33': ["MSSCFormerPlansv2.3"],
        'MSSCFormerTrainerV2_O2': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_ResencUNet': ["MSSCFormerPlans_FabiansResUNet_v2.1"],
        'MSSCFormerTrainerV2_DA2': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_allConv3x3': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_ForceBD': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_ForceSD': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_LReLU_slope_2en1': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_lReLU_convReLUIN': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_ReLU': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_ReLU_biasInSegOutput': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_ReLU_convReLUIN': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_lReLU_biasInSegOutput': ["MSSCFormerPlansv2.1"],
        #'MSSCFormerTrainerV2_Loss_MCC': ["MSSCFormerPlansv2.1"],
        #'MSSCFormerTrainerV2_Loss_MCCnoBG': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_Loss_DicewithBG': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_Loss_Dice_LR1en3': ["MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_Loss_Dice': ["MSSCFormerPlans", "MSSCFormerPlansv2.1"],
        'MSSCFormerTrainerV2_Loss_DicewithBG_LR1en3': ["MSSCFormerPlansv2.1"],
        # 'MSSCFormerTrainerV2_fp32': ["MSSCFormerPlansv2.1"],
        # 'MSSCFormerTrainerV2_fp32': ["MSSCFormerPlansv2.1"],
        # 'MSSCFormerTrainerV2_fp32': ["MSSCFormerPlansv2.1"],
        # 'MSSCFormerTrainerV2_fp32': ["MSSCFormerPlansv2.1"],
        # 'MSSCFormerTrainerV2_fp32': ["MSSCFormerPlansv2.1"],

    }

    trainers = ['MSSCFormerTrainer'] + ['MSSCFormerTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'MSSCFormerTrainerNewCandidate24_2',
        'MSSCFormerTrainerNewCandidate24_3',
        'MSSCFormerTrainerNewCandidate26_2',
        'MSSCFormerTrainerNewCandidate27_2',
        'MSSCFormerTrainerNewCandidate23_always3DDA',
        'MSSCFormerTrainerNewCandidate23_corrInit',
        'MSSCFormerTrainerNewCandidate23_noOversampling',
        'MSSCFormerTrainerNewCandidate23_softDS',
        'MSSCFormerTrainerNewCandidate23_softDS2',
        'MSSCFormerTrainerNewCandidate23_softDS3',
        'MSSCFormerTrainerNewCandidate23_softDS4',
        'MSSCFormerTrainerNewCandidate23_2_fp16',
        'MSSCFormerTrainerNewCandidate23_2',
        'MSSCFormerTrainerVer2',
        'MSSCFormerTrainerV2_2',
        'MSSCFormerTrainerV2_3',
        'MSSCFormerTrainerV2_3_CE_GDL',
        'MSSCFormerTrainerV2_3_dcTopk10',
        'MSSCFormerTrainerV2_3_dcTopk20',
        'MSSCFormerTrainerV2_3_fp16',
        'MSSCFormerTrainerV2_3_softDS4',
        'MSSCFormerTrainerV2_3_softDS4_clean',
        'MSSCFormerTrainerV2_3_softDS4_clean_improvedDA',
        'MSSCFormerTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'MSSCFormerTrainerV2_3_softDS4_radam',
        'MSSCFormerTrainerV2_3_softDS4_radam_lowerLR',

        'MSSCFormerTrainerV2_2_schedule',
        'MSSCFormerTrainerV2_2_schedule2',
        'MSSCFormerTrainerV2_2_clean',
        'MSSCFormerTrainerV2_2_clean_improvedDA_newElDef',

        'MSSCFormerTrainerV2_2_fixes', # running
        'MSSCFormerTrainerV2_BN', # running
        'MSSCFormerTrainerV2_noDeepSupervision', # running
        'MSSCFormerTrainerV2_softDeepSupervision', # running
        'MSSCFormerTrainerV2_noDataAugmentation', # running
        'MSSCFormerTrainerV2_Loss_CE', # running
        'MSSCFormerTrainerV2_Loss_CEGDL',
        'MSSCFormerTrainerV2_Loss_Dice',
        'MSSCFormerTrainerV2_Loss_DiceTopK10',
        'MSSCFormerTrainerV2_Loss_TopK10',
        'MSSCFormerTrainerV2_Adam', # running
        'MSSCFormerTrainerV2_Adam_MSSCFormerTrainerlr', # running
        'MSSCFormerTrainerV2_SGD_ReduceOnPlateau', # running
        'MSSCFormerTrainerV2_SGD_lr1en1', # running
        'MSSCFormerTrainerV2_SGD_lr1en3', # running
        'MSSCFormerTrainerV2_fixedNonlin', # running
        'MSSCFormerTrainerV2_GeLU', # running
        'MSSCFormerTrainerV2_3ConvPerStage',
        'MSSCFormerTrainerV2_NoNormalization',
        'MSSCFormerTrainerV2_Adam_ReduceOnPlateau',
        'MSSCFormerTrainerV2_fp16',
        'MSSCFormerTrainerV2', # see overwrite_plans
        'MSSCFormerTrainerV2_noMirroring',
        'MSSCFormerTrainerV2_momentum09',
        'MSSCFormerTrainerV2_momentum095',
        'MSSCFormerTrainerV2_momentum098',
        'MSSCFormerTrainerV2_warmup',
        'MSSCFormerTrainerV2_Loss_Dice_LR1en3',
        'MSSCFormerTrainerV2_NoNormalization_lr1en3',
        'MSSCFormerTrainerV2_Loss_Dice_squared',
        'MSSCFormerTrainerV2_newElDef',
        'MSSCFormerTrainerV2_fp32',
        'MSSCFormerTrainerV2_cycleAtEnd',
        'MSSCFormerTrainerV2_reduceMomentumDuringTraining',
        'MSSCFormerTrainerV2_graduallyTransitionFromCEToDice',
        'MSSCFormerTrainerV2_insaneDA',
        'MSSCFormerTrainerV2_independentScalePerAxis',
        'MSSCFormerTrainerV2_Mish',
        'MSSCFormerTrainerV2_Ranger_lr3en4',
        'MSSCFormerTrainerV2_cycleAtEnd2',
        'MSSCFormerTrainerV2_GN',
        'MSSCFormerTrainerV2_DP',
        'MSSCFormerTrainerV2_FRN',
        'MSSCFormerTrainerV2_resample33',
        'MSSCFormerTrainerV2_O2',
        'MSSCFormerTrainerV2_ResencUNet',
        'MSSCFormerTrainerV2_DA2',
        'MSSCFormerTrainerV2_allConv3x3',
        'MSSCFormerTrainerV2_ForceBD',
        'MSSCFormerTrainerV2_ForceSD',
        'MSSCFormerTrainerV2_ReLU',
        'MSSCFormerTrainerV2_LReLU_slope_2en1',
        'MSSCFormerTrainerV2_lReLU_convReLUIN',
        'MSSCFormerTrainerV2_ReLU_biasInSegOutput',
        'MSSCFormerTrainerV2_ReLU_convReLUIN',
        'MSSCFormerTrainerV2_lReLU_biasInSegOutput',
        'MSSCFormerTrainerV2_Loss_DicewithBG_LR1en3',
        #'MSSCFormerTrainerV2_Loss_MCCnoBG',
        'MSSCFormerTrainerV2_Loss_DicewithBG',
        # 'MSSCFormerTrainerV2_Loss_Dice_LR1en3',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
        # 'MSSCFormerTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
