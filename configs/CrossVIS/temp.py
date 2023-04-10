MODEL:
  META_ARCHITECTURE: "CrossVIS"
  MASK_ON: True
  BACKBONE:
    NAME: "build_fcos_resnet_fpn_backbone"
  RESNETS:
    OUT_FEATURES: ["res3", "res4", "res5"]
  FPN:
    IN_FEATURES: ["res3", "res4", "res5"]
  PROPOSAL_GENERATOR:
    NAME: "FCOS"
  FCOS:
    THRESH_WITH_CTR: True
    USE_SCALE: True
    NUM_CLASSES: 40
  CONDINST:
    MAX_PROPOSALS: 500
  CROSSVIS_ON: True
DATASETS:
  TRAIN: ("youtubevis_train",)
  TEST: ("youtubevis_valid",)
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.005
  STEPS: (15000, 21000)
  MAX_ITER: 23000
  CHECKPOINT_PERIOD: 5
INPUT:
  MIN_SIZE_TRAIN: (360,)
  MAX_SIZE_TRAIN: 640
  MIN_SIZE_TEST: 360
  MAX_SIZE_TEST: 640
