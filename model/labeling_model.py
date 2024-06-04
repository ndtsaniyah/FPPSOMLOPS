from pyabsa import ABSADatasetList, APCModelList, available_checkpoints
from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import ATEPCConfigManager

config = ATEPCConfigManager.get_atepc_config_english()
dataset = ABSADatasetList.SemEval

checkpoint_map = available_checkpoints(
    TaskCodeOption.Aspect_Polarity_Classification, show_ckpts=True
)

aspect_extractor = ATEPC.ATEPCTrainer(
    config=config,
    from_checkpoint=checkpoint_map,   # not necessary for most situations
    dataset=chinese_sets,
    checkpoint_save_mode=1,
    auto_device=True,
    load_aug=False,
).load_trained_model()

def label_text(text):
    # Implementasi pelabelan dengan model yang sudah dilatih
    result = aspect_extractor.infer(text)
    return result