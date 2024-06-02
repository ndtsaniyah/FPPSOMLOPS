from pyabsa import ABSADatasetList, APCModelList, Trainer
from pyabsa import ATEPCConfigManager

config = ATEPCConfigManager.get_atepc_config_english()
dataset = ABSADatasetList.SemEval
sent_classifier = Trainer(config=config,
                          dataset=dataset,
                          model=APCModelList.FAST_LCF_ATEPC,
                          save_checkpoint=True,
                          auto_device=True)

def label_text(text):
    # Implementasi pelabelan dengan model yang sudah dilatih
    result = sent_classifier.infer(text)
    return result
