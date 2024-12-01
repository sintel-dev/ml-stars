from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch.loggers import WandbLogger
import pytorch_lightning as pl
import wandb as wandb
from torch.optim import Adam
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import csv

TRAIN_PCT = 0.7
VAL_PCT = 0.85
DATA_PATH = "/mnt/zephyr/modeling/tft_format.csv"
DF = pd.read_csv(DATA_PATH, 
            header=0, 
            date_format="%Y-%m-%d %H:%M:%S", 
            parse_dates=True,
            keep_date_col=False)
COLUMN_NAMES = ["time_idx","time","label","turbine","WGEN_IntTmp_max","WGEN_IntTmp_mean","WGEN_IntTmp_min","WGEN_IntTmp_sd",'WGEN_ClSt_max',"WGEN_ClSt_mean",'WGEN_ClSt_min',"WCNV_Torq_mean","WCNV_Torq_max","WGEN_Spd_max","WGEN_Spd_mean","WGEN_Spd_min","WGEN_W_mean","WGEN_W_max","WCNV_Torq_sd","WGEN_Spd_sd","WGEN_W_min","WGEN_W_sd","WTOW_PlatTmp_mean","WNAC_WdSpd1_mean","WNAC_WdSpd1_max","WNAC_WdSpd2_max","WNAC_WdSpd2_mean","WNAC_WdDir1_mean","WNAC_WdDir2_mean","WNAC_ExtPres_mean","WNAC_WdSpdAvg_mean","WNAC_WdSpdAvg_max","WNAC_WdSpdAvg_min","WNAC_IntTmp_mean","WNAC_IntTmp_max","WNAC_IntTmp_min",'WNAC_IntTmp_sd',"WNAC_Vib1_max","WNAC_Vib1_min",'WNAC_Vib1_mean','WNAC_Vib2_max',"WNAC_Vib2_mean","WNAC_Vib2_min","WNAC_Vib3_max","WNAC_Vib3_mean","WNAC_Vib3_min","WNAC_Vib4_max","WNAC_Vib4_mean","WNAC_Vib4_min","WTRM_FtrPres1_mean","WTRM_FtrPres1_max","WTRM_FtrPres2_max","WTRM_FtrPres2_mean",'WTRM_FtrPres1_min',"WTRM_FtrPres2_min","WROT_Spd1_mean","WROT_Spd1_max","WROT_Spd1_min","WROT_Spd1_sd","WROT_Spd2_max","WROT_Spd2_mean","WROT_Spd2_min","WROT_Spd2_sd","WROT_Pos_max","WROT_Pos_mean","WROT_Pos_min","WROT_MnBrgTemp1_max","WROT_MnBrgTemp1_mean","WROT_MnBrgTemp1_min","WROT_MnBrgTemp2_max","WROT_MnBrgTemp2_mean","WROT_MnBrgTemp2_min","WTRM_HyFtrPres1_mean","WTRM_HyFtrPres1_max","WTRM_HyFtrPres1_min","WTRM_HyFtrPres1_sd","WTRM_HyFtrPres2_max","WTRM_HyFtrPres2_mean","WTRM_HyFtrPres2_min","WTRM_HyFtrPres2_sd","WTRM_HySysPres1_max","WTRM_HySysPres1_mean","WTRM_HySysPres1_min","WTRM_HySysLockPres1_max","WTRM_HySysLockPres1_mean","WTRM_HySysLockPres1_sd","WTRM_HySysLockPres1_min","WROT_LockPos1_mean","WROT_LockPos1_max","WROT_LockPos1_min","WROT_LockPos1_sd","WROT_LockPos2_max","WROT_LockPos2_mean","WROT_LockPos2_min","WROT_LockPos2_sd","WROT_LockPos3_max","WROT_LockPos3_mean","WROT_LockPos3_min","WROT_LockPos3_sd","WROT_Brk2HyTmp6_sd","WROT_Brk2HyTmp6_min","WROT_Brk2HyTmp6_mean","WROT_Brk2HyTmp6_max","WROT_Brk2HyTmp5_sd","WROT_Brk2HyTmp5_min","WROT_Brk2HyTmp5_mean","WROT_Brk2HyTmp5_max","WROT_Brk1HyTmp6_min","WROT_Brk1HyTmp6_sd","WROT_Brk1HyTmp6_mean","WROT_Brk1HyTmp6_max","WROT_Brk1HyTmp5_sd","WROT_Brk1HyTmp5_min","WROT_Brk1HyTmp5_mean","WROT_Brk1HyTmp5_max","WROT_Brk2HyTmp4_sd","WROT_Brk2HyTmp4_min","WROT_Brk2HyTmp4_max","WROT_Brk2HyTmp3_sd","WROT_Brk2HyTmp4_mean","WROT_Brk2HyTmp3_min","WROT_Brk2HyTmp3_mean","WROT_Brk2HyTmp3_max","WROT_Brk1HyTmp4_sd","WROT_Brk1HyTmp4_min","WROT_Brk1HyTmp4_mean","WROT_Brk1HyTmp4_max","WROT_Brk1HyTmp3_sd","WROT_Brk1HyTmp3_min","WROT_Brk1HyTmp3_mean","WROT_Brk1HyTmp3_max","WROT_Brk2HyTmp2_sd","WROT_Brk2HyTmp2_min","WROT_Brk2HyTmp2_mean","WROT_Brk2HyTmp2_max","WROT_HyOilTmp1_sd","WROT_HyOilTmp1_min","WROT_HyOilTmp1_mean","WROT_HyOilTmp1_max","WROT_Brk2HyTmp1_sd","WROT_Brk2HyTmp1_min","WROT_Brk2HyTmp1_mean","WROT_Brk2HyTmp1_max","WROT_Brk1HyTmp2_sd","WROT_Brk1HyTmp2_min","WROT_Brk1HyTmp2_mean","WROT_Brk1HyTmp2_max","WROT_Brk1HyTmp1_sd","WROT_Brk1HyTmp1_min","WROT_Brk1HyTmp1_mean","WROT_Brk1HyTmp1_max","WROT_Brk1HyPres_max","WROT_Brk1HyPres_mean","WROT_Brk1HyPres_min","WROT_Brk1HyPres_sd","WROT_Brk2HyPres_max","WROT_Brk2HyPres_mean","WROT_Brk2HyPres_min","WROT_Brk2HyPres_sd","WROT_Brk1HyAccPres_max","WROT_Brk1HyAccPres_mean","WROT_Brk1HyAccPres_min","WROT_Brk1HyAccPres_sd","WROT_Brk2HyAccPres_max","WROT_Brk2HyAccPres_mean","WROT_Brk2HyAccPres_min","WROT_Brk2HyAccPres_sd","groupconst"]

def format_data(DF: pd.DataFrame):
    # Copy dataframe
    df = DF

    # Ensure 'turbine' is a categorical column, turn 'label' from bool to float
    df['label'] = df['label'].astype(float)
    df['turbine'] = df['turbine'].astype('category')

    # Create a 'time_idx' column that resets for each unique 'group_id'
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values(by=['time', 'turbine'])
    df.insert(0, 'time_idx', df.groupby('turbine').cumcount())
    df['groupconst'] = 0

    # DEBUGGING
    f = open("/home/boom90lb/Zephyr/tftmodel/tft_formatted.csv", "w")
    f.write(df.to_csv(None, index=False))
    f.close()

    training_cutoff = np.floor(df['time_idx'].max()*TRAIN_PCT)
    validation_cutoff = np.floor(df['time_idx'].max()*VAL_PCT)

    return df, training_cutoff, validation_cutoff

def get_column_names(data_path="."):
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            return row

TSDS_PARAMS = {
    "time_idx": "time_idx",
    "target": "label",
    "group_ids": ["turbine"],
    "min_encoder_length": 1,
    "max_encoder_length": 16,
    "min_prediction_length": 1,
    "max_prediction_length": 8,
    "static_categoricals": [],
    "static_reals": ["groupconst"],
    "time_varying_known_reals": [],
    "time_varying_unknown_reals": COLUMN_NAMES[6:-2] + ["label"],
    "time_varying_unknown_categoricals": [],
    "allow_missing_timesteps": True,
}

TFT_PARAMS = {
    "hidden_size": 16,
    "lstm_layers": 2,
    "dropout": 0.2,
    "output_size": 1,
    "loss": RMSE(),
    "attention_head_size": 4,
    "max_encoder_length": 32,
    "allowed_encoder_known_variable_names": COLUMN_NAMES[6:-2],
    "hidden_continuous_size": 8, 
    "learning_rate": 0.005, 
    "log_interval": 10,
    "optimizer": "Adam",
    "log_val_interval": 1,  
    "reduce_on_plateau_patience": 4,
    "monotone_constaints": {},
    "share_single_variable_networks": False,
    "causal_attention": True,
}

TRAINER_PARAMS = {
    "max_epochs": 10,
    "accelerator": "auto"
}

class TFTLightningModule(pl.LightningModule):
    def __init__(self, df, **kwargs):
        super().__init__()
        self.model = TemporalFusionTransformer.from_dataset(df, **kwargs)
        self.automatic_optimization = False
        self.optimizers = self.configure_optimizers()
        self.criterion = RMSE()
        wandb.save_hyperparameters()
        

    def forward(self, x):
        wandb.log('x_out', self.model.forward(x))
        return self.model.forward(x)
    
    def training_step(self, batch):
        opt: Adam = self.optimizers
        x, y = batch

        opt.zero_grad()

        out = self.forward(x)
        loss = self.criterion.loss(out, y)

        self.model.backward(loss)
        opt.step()
        
        wandb.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-3)
    
def train(training_data: TimeSeriesDataSet, validation_data: TimeSeriesDataSet):
    model2 = TFTLightningModule(training_data, **TFT_PARAMS)
    wandb.watch(model2)
    training_loader = training_data.to_dataloader(train=True, batch_size=8, num_workers=4)
    validation_loader = validation_data.to_dataloader(train=False, batch_size=8, num_workers=4)

    wandb_logger = WandbLogger(log_model="all")
    trainer = pl.Trainer(logger=wandb_logger, **TRAINER_PARAMS)
    trainer.fit(model2, training_loader, validation_loader)

    return trainer
    
def eval(training_data):
    frozen_model = TFTLightningModule(training_data, **TFT_PARAMS)

    frozen_model.load_state_dict(torch.load("/home/boom90lb/Zephyr/tftmodel/checkpoints/saves/11-23-2023;22-49-11.ckpt"))

    frozen_model.eval()

    training_dataloader = training_data.to_dataloader(False, 4)

    loss_running_sum = 0
    num_batches = 0

    for batch, i in enumerate(training_dataloader):
        x, y = batch
        out = frozen_model.forward(x)
        loss_fn = RMSE()
        
        loss = loss_fn.loss(out, y)
        print(loss)
        loss_running_sum += loss

        if i > num_batches:
            num_batches = i

    return loss_running_sum/num_batches


def main():
    df, training_cutoff, validation_cutoff = format_data(DF)

    training_data = TimeSeriesDataSet(df.loc[df['time_idx'] <= training_cutoff], **TSDS_PARAMS)

    validation_data = TimeSeriesDataSet(df.loc[df['time_idx'] <= validation_cutoff], **TSDS_PARAMS, min_prediction_idx=training_cutoff + 1)

    test_data = TimeSeriesDataSet(df, **TSDS_PARAMS, min_prediction_idx=validation_cutoff + 1)

    # prev_eval = eval(training_data)

    trained_model = train(training_data, validation_data)

    trained_model.save_checkpoint("/home/boom90lb/Zephyr/tftmodel/checkpoints/model" + datetime.now().strftime("%m-%d-%Y;%H-%M-%S") + ".ckpt")

    torch.save(trained_model.model.state_dict(), "/home/boom90lb/Zephyr/tftmodel/checkpoints/saves/" + datetime.now().strftime("%m-%d-%Y;%H-%M-%S") + ".ckpt")
    
    wandb.finish()

if __name__ == "__main__":
    main()