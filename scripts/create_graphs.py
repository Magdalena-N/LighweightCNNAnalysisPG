import os.path
import pandas as pd
import plotnine as p9
import inspect
from typing import Dict
from plotnine import *
from pathlib import Path
import numpy as np
import time

p9.theme_set(p9.theme_bw(base_size=18))

CSV_PATH = "D:/Dokumenty/Mgr - CNNs/Aplikacje/TFLiteAndroidApp/python/scripts/Data preparation/classification/results_csv/"
OUTPUT_PATH = 'output'
PLOT_PATH = os.path.join(OUTPUT_PATH, 'plots')


# summarize_data = "Results_summarized_20240810-114534.csv"


# summarize_data = "Results_summarized_20240712-164920.csv"


def _retrieve_name(var):
    for fi in reversed(inspect.stack()):
        for name in [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]:
            if name != 'ggplot':
                return name


def save_gg_plot(gg_plot: ggplot, filename: str = None, aspect_ration: float = 1.0):
    filename = _retrieve_name(gg_plot) if filename is None else filename
    Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
    gg_plot.save('.'.join([os.path.join(PLOT_PATH, filename), 'svg']), width=18 * aspect_ration,
                 height=12 * aspect_ration)  # height changed from 8.27


def prepare_plot(plot: p9.ggplot, filename: str, x_lab: str, y_lab: str, angle: int = 0):
    plot += p9.theme(axis_text_x=p9.element_text(angle=angle, hjust=(90 - angle / 2) / 90))
    # plot += p9.scale_fill_brewer(palette="Greens")
    if x_lab is not None:
        plot += p9.xlab(x_lab)

    if y_lab is not None:
        plot += p9.ylab(y_lab)

    save_gg_plot(plot, filename)


def create_col_plot(df: pd.DataFrame, x: str, y: str, filename: str, x_lab: str, y_lab: str, angle: int = 0, y_lim=None,
                    order=None):
    col_plot = p9.ggplot(df, p9.aes(x=x, y=y, fill="Batch size")) + p9.geom_col(fill="steelblue", position='dodge')
    if y_lim is not None:
        col_plot += p9.coord_cartesian(ylim=(y_lim[0], y_lim[1]))
    if order is not None:
        col_plot += p9.scale_x_discrete(limits=order)
    prepare_plot(col_plot, filename, x_lab, y_lab, angle)


def split_dataframe(df) -> pd.DataFrame:
    df[["name", "Model version", "alpha", "rho", "type"]] = df["Net Model"].str.split('_', n=4, expand=True)

    df["type"] = df["type"].str.replace(r'_*uint8', '', regex=True)
    df["type"] = df["type"].str.replace(r'_*quant', '', regex=True)
    df["type"] = df["type"].replace(r'^\s*$', None, regex=True)

    df["alpha"] = df["Net Model"].str.split('_', expand=True)[2]
    df["rho"] = df["Net Model"].str.split('_', expand=True)[3]
    df["Model version label"] = df[["Model version", 'type']].apply(
        lambda x: '_'.join([e for e in x if pd.notnull(e)]), axis=1)
    df["Model version"] = df[["Model version", 'type']].apply(
        lambda x: '_'.join([e.split('_')[0] for e in x if pd.notnull(e)]), axis=1)
    df["label"] = df["Model version label"] + "_" + df["alpha"] + "_" + \
                  df["rho"]

    df["Precision"] = "float32"
    df.loc[df["Net Model"].str.contains("quant"), 'Precision'] = 'uint8'
    df.loc[df["Net Model"].str.contains("uint8"), 'Precision'] = 'uint8'

    # df.to_csv(
    #     f'D:/Dokumenty/Mgr - CNNs/Aplikacje/TFLiteAndroidApp/python/scripts/Data preparation/classification/results_csv/Results_split_{time.strftime("%Y%m%d-%H%M%S")}.csv',
    #     index=False)

    return df


def prepare_dataframe(summarize_data=None, type='Classification') -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(os.path.join(CSV_PATH, summarize_data))
    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds
    # rename mobilenet_96 to mobilenet_096
    df = df.replace("_96", "_096", regex=True)
    # rename mobilenet_v3_1.0_224_uint8 to mobilenet_v3_1.0_224_large_uint8
    df = df.replace("mobilenet_v3_1.0_224_uint8", "mobilenet_v3_1.0_224_large_uint8", regex=True)

    split_dataframe(df)

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df
    # df_dict['default_models'] = df.loc[~df["Net Model"].str.contains("quant") & ~df["Net Model"].str.contains("uint8")]
    # df_dict['quant_models'] = df.loc[df["Net Model"].str.contains("quant") & df["Net Model"].str.contains("uint8")]
    #
    # df_dict['s24_def'] = df_dict['default_models'].loc[df_dict['default_models']['Phone'] == 'Samsung Galaxy S24']
    #
    # df_dict['s24_all'] = df_dict['original'].loc[df_dict['original']['Phone'] == 'Samsung Galaxy S24']
    # df_dict['s24_all'] = df_dict['s24_all'].loc[df_dict['s24_all']['Batch size'] == 1]
    # df_dict['s24_all'][["name", "Model version", "alpha", "rho", "type"]] = df_dict['s24_all']["Net Model"].str.split(
    #     '_', n=4, expand=True)
    # df_dict['s24_all']["type"] = df_dict['s24_all']["type"].str.replace(r'_*uint8', '', regex=True)
    # df_dict['s24_all']["type"] = df_dict['s24_all']["type"].str.replace(r'_*quant', '', regex=True)
    # df_dict['s24_all']["type"] = df_dict['s24_all']["type"].replace(r'^\s*$', None, regex=True)
    #
    # df_dict['s24_all']["alpha"] = df_dict['s24_all']["Net Model"].str.split('_', expand=True)[2]
    # df_dict['s24_all']["rho"] = df_dict['s24_all']["Net Model"].str.split('_', expand=True)[3]
    # df_dict["s24_all"]["Model version"] = df_dict["s24_all"][["Model version", 'type']].apply(
    #     lambda x: '_'.join([e for e in x if pd.notnull(e)]), axis=1)
    # df_dict['s24_all']["label"] = df_dict["s24_all"]["Model version"] + "_" + df_dict["s24_all"]["alpha"] + "_" + \
    #                               df_dict["s24_all"]["rho"]
    #
    # df_dict['s24_all']["Precision"] = "float32"
    # df_dict["s24_all"].loc[df_dict['s24_all']["Net Model"].str.contains("quant"), 'Precision'] = 'uint8'
    # df_dict["s24_all"].loc[df_dict['s24_all']["Net Model"].str.contains("uint8"), 'Precision'] = 'uint8'
    # df_dict["s24_all"].to_csv(
    #     f'D:/Dokumenty/Mgr - CNNs/Aplikacje/objectdetection/python/scripts/Data preparation/classification/results_csv/Results_split_{time.strftime("%Y%m%d-%H%M%S")}.csv',
    #     index=False)
    #
    # # S24 for different mobilenet
    # df_dict['s24_mv1_def'] = df_dict['s24_def'].loc[df_dict['s24_def']['Net Model'].str.contains("v1")]
    # df_dict['s24_mv2_def'] = df_dict['s24_def'].loc[df_dict['s24_def']['Net Model'].str.contains("v2")]
    # df_dict['s24_mv3_def'] = df_dict['s24_def'].loc[df_dict['s24_def']['Net Model'].str.contains("v3")]

    # S24 for mobilenet v1 for CPU
    # df_dict['s24_mv1_CPU_def'] = df_dict['s24_mv1_def'].loc[df_dict['s24_mv1_def']['Delegate'] == 'CPU']
    # df_dict['s24_mv1_CPU_def']["Model Median per image"] = df_dict['s24_mv1_CPU_def']["Model Median"] / \
    #                                                        df_dict['s24_mv1_CPU_def']["Batch size"]
    # df_dict['s24_mv1_CPU_def']["Model Median per image"] = df_dict['s24_mv1_CPU_def']["Model Median per image"].astype(
    #     int)
    # df_dict['s24_mv1_CPU_def']['Batch size'] = df_dict['s24_mv1_CPU_def']['Batch size'].apply(str)
    # df_dict['s24_mv1_CPU_def']['Batch size'] = df_dict['s24_mv1_CPU_def']['Batch size'].str.zfill(2)

    #
    # # S24 for different batch size for mv1
    # df_dict['s24_mv1_bs1_def'] = df_dict['s24_mv1_CPU_def'].loc[df_dict['s24_mv1_CPU_def']['Batch size'] == '1']
    # df_dict['s24_mv1_bs2_def'] = df_dict['s24_mv1_CPU_def'].loc[df_dict['s24_mv1_CPU_def']['Batch size'] == '2']
    # df_dict['s24_mv1_bs4_def'] = df_dict['s24_mv1_CPU_def'].loc[df_dict['s24_mv1_CPU_def']['Batch size'] == '4']
    # df_dict['s24_mv1_bs8_def'] = df_dict['s24_mv1_CPU_def'].loc[df_dict['s24_mv1_CPU_def']['Batch size'] == '8']
    # df_dict['s24_mv1_bs16_def'] = df_dict['s24_mv1_CPU_def'].loc[df_dict['s24_mv1_CPU_def']['Batch size'] == '16']
    # df_dict['s24_mv1_bs32_def'] = df_dict['s24_mv1_CPU_def'].loc[df_dict['s24_mv1_CPU_def']['Batch size'] == '32']

    return df_dict


def compare_delegates(df_dict):
    df_dict["createdAtNotNull"] = df_dict["default_models"].loc[df_dict["default_models"]["CreatedAt"].notnull()]
    df_dict["s24_notnull"] = df_dict["createdAtNotNull"].loc[
        df_dict["createdAtNotNull"]["Phone"] == 'Samsung Galaxy S24']
    # df_dict["s24_notnull_CPU"] = df_dict['s24_notnull'].loc[df_dict['s24_notnull']['Delegate'] == 'CPU']
    df_dict["s24_notnull_20240807"] = df_dict['s24_notnull'].loc[
        df_dict['s24_notnull']['CreatedAt'].str.contains("2024-08-07")]
    df_dict["s24_notnull_20240810"] = df_dict['s24_notnull'].loc[
        df_dict['s24_notnull']['CreatedAt'].str.contains("2024-08-10")]
    df_dict["s24_notnull_20240807_mv1"] = df_dict["s24_notnull_20240807"].loc[
        df_dict['s24_notnull_20240807']['Model version'] == "v1"]
    df_dict["s24_notnull_20240810_mv2"] = df_dict["s24_notnull_20240810"].loc[
        df_dict['s24_notnull_20240810']['Model version'] == "v2"]
    df_dict["s24_notnull_20240807_mv3"] = df_dict["s24_notnull_20240807"].loc[
        df_dict['s24_notnull_20240807']['Model version'].str.contains("v3")]

    # CPU vs GPU vs NNAPI s24 all models
    # model_delay_s24_CPUvsGPUvsNNAPI = p9.ggplot(df_dict['s24_notnull_20240807'],
    #                                             p9.aes(x='Net Model', y='Model Median'))
    # model_delay_s24_CPUvsGPUvsNNAPI += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    # model_delay_s24_CPUvsGPUvsNNAPI += p9.scale_fill_discrete(name=" ")
    # prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI,
    #              f'default_models_s24_GPUvsCPUvsNNAPI_{time.strftime("%Y%m%d-%H%M%S")}', 'Model',
    #              'Czas inferencji [ns]',
    #              angle=90)

    # CPU vs GPU vs NNAPI s24 mv1
    model_delay_s24_CPUvsGPUvsNNAPI_v1 = p9.ggplot(df_dict['s24_notnull_20240807_mv1'],
                                                   p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsNNAPI_v1 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsNNAPI_v1 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v1, f'mv1_s24_GPUvsCPUvsNNAPI_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ns]',
                 angle=90)

    # CPU vs GPU vs NNAPI s24 mv2
    model_delay_s24_CPUvsGPUvsNNAPI_v2 = p9.ggplot(df_dict['s24_notnull_20240810_mv2'],
                                                   p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsNNAPI_v2 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsNNAPI_v2 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v2, f'mv2_s24_GPUvsCPUvsNNAPI_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ns]',
                 angle=90)

    # CPU vs GPU vs NNAPI s24 mv3
    model_delay_s24_CPUvsGPUvsNNAPI_v3 = p9.ggplot(df_dict['s24_notnull_20240807_mv3'],
                                                   p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsNNAPI_v3 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsNNAPI_v3 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v3, f'mv3_s24_GPUvsCPUvsNNAPI_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ns]',
                 angle=90)


# def compare_delegates_2(df_dict):
#     # CPU vs GPU vs NNAPI s24 all models
#     model_delay_s24_CPUvsGPUvsNNAPI = p9.ggplot(df_dict['s24_def'], p9.aes(x='Net Model', y='Model Median'))
#     model_delay_s24_CPUvsGPUvsNNAPI += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
#     model_delay_s24_CPUvsGPUvsNNAPI += p9.scale_fill_discrete(name=" ")
#     prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI, '5_default_models_s24_GPUvsCPUvsNNAPI', 'Model',
#                  'Czas inferencji [ns]',
#                  angle=90)
#
#     # CPU vs GPU vs NNAPI s24 mv1
#     model_delay_s24_CPUvsGPUvsNNAPI_v1 = p9.ggplot(df_dict['s24_mv1_def'], p9.aes(x='Net Model', y='Model Median'))
#     model_delay_s24_CPUvsGPUvsNNAPI_v1 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
#     model_delay_s24_CPUvsGPUvsNNAPI_v1 += p9.scale_fill_discrete(name=" ")
#     prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v1, '5_mv1_s24_GPUvsCPUvsNNAPI', 'Model',
#                  'Czas inferencji [ns]',
#                  angle=90)
#
#     # CPU vs GPU vs NNAPI s24 mv2
#     model_delay_s24_CPUvsGPUvsNNAPI_v2 = p9.ggplot(df_dict['s24_mv2_def'], p9.aes(x='Net Model', y='Model Median'))
#     model_delay_s24_CPUvsGPUvsNNAPI_v2 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
#     model_delay_s24_CPUvsGPUvsNNAPI_v2 += p9.scale_fill_discrete(name=" ")
#     prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v2, '5_mv2_s24_GPUvsCPUvsNNAPI', 'Model',
#                  'Czas inferencji [ns]',
#                  angle=90)
#
#     # CPU vs GPU vs NNAPI s24 mv3
#     model_delay_s24_CPUvsGPUvsNNAPI_v3 = p9.ggplot(df_dict['s24_mv3_def'], p9.aes(x='Net Model', y='Model Median'))
#     model_delay_s24_CPUvsGPUvsNNAPI_v3 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
#     model_delay_s24_CPUvsGPUvsNNAPI_v3 += p9.scale_fill_discrete(name=" ")
#     prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v3, '5_mv3_s24_GPUvsCPUvsNNAPI', 'Model',
#                  'Czas inferencji [ns]',
#                  angle=90)


def compare_bs_mv1():
    # summarize_data = "Results_summarized_20240829-184554.csv"
    # summarize_data = "Results_summarized_20240829-184554_changed_mv1.csv"
    # summarize_data = "Results_summarized_20240712-164920_bs_CPU1_add_other_rho.csv"
    summarize_data = "Results_summarized_20240829-202402.csv"
    df_dict = prepare_dataframe(summarize_data)

    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Precision"] == "float32"]

    df_dict["mv1"] = df_dict["original"].loc[df_dict['original']['Model version'] == "v1"]
    df_dict["mv1"]["Model Median per Image"] = df_dict['mv1']["Model Median"] / df_dict['mv1']["Batch size"]
    # df_dict['mv1']["Model Median per Image"] = df_dict['mv1']["Model Median per Image"].astype(int)
    df_dict['mv1']["Batch size"] = pd.Categorical(df_dict['mv1']["Batch size"])

    plot_df = df_dict["mv1"]

    plot = p9.ggplot(plot_df, p9.aes(x='Net Model', y='Model Median per Image', fill='Batch size'))
    plot += p9.geom_col(stat='identity', position='dodge')
    prepare_plot(plot, f'mv1_s24_CPU5_bs_{time.strftime("%Y%m%d-%H%M%S")}', 'Model', 'Czas inferencji [ms]', angle=90)


def compare_bs_mv1_1_0_224(phone="OnePlus 9 Pro"):
    # lista plikow dla s24
    # files = ["Results_summarized_20240712-164920_bs_CPU1.csv", "Results_summarized_20240829-202402.csv",
    #          "Results_summarized_20240829-184554_changed_mv1.csv"]

    files = ['Results_summarized_20240907-130256.csv']
    li = []

    for filename in files:
        df_tmp = pd.read_csv(os.path.join(CSV_PATH, filename), index_col=None, header=0)
        li.append(df_tmp)

    df = pd.concat(li, axis=0, ignore_index=True)

    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds

    # rename mobilenet_96 to mobilenet_096
    df = df.replace("_96", "_096", regex=True)
    # rename mobilenet_v3_1.0_224_uint8 to mobilenet_v3_1.0_224_large_uint8
    df = df.replace("mobilenet_v3_1.0_224_uint8", "mobilenet_v3_1.0_224_large_uint8", regex=True)

    split_dataframe(df)

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df
    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Phone"] == phone]
    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Precision"] == "float32"]
    df_dict['original'] = df_dict['original'].loc[df_dict['original']["Net Model"].str.contains("1.0_224")]

    df_dict["mv1"] = df_dict["original"].loc[df_dict['original']['Model version'] == "v1"]
    df_dict["mv1"]["Model Median per Image"] = df_dict['mv1']["Model Median"] / df_dict['mv1']["Batch size"]
    # df_dict['mv1']["Model Median per Image"] = df_dict['mv1']["Model Median per Image"].astype(int)
    df_dict['mv1']["Batch size"] = pd.Categorical(df_dict['mv1']["Batch size"])

    plot_df = df_dict["mv1"]

    plot = p9.ggplot(plot_df, p9.aes(x='Delegate', y='Model Median per Image', fill='Batch size'))
    plot += p9.geom_col(stat='identity', position='dodge')
    prepare_plot(plot, f'6_3_2_{phone}_bs_1_0_224_{time.strftime("%Y%m%d-%H%M%S")}', 'Delegat',
                 'Czas inferencji [ms]', angle=0)


def compare_bs_mv2():
    # summarize_data = "Results_summarized_20240829-184554.csv"
    # summarize_data = "Results_summarized_20240712-164920_bs_CPU1_add_other_rho.csv"
    summarize_data = "Results_summarized_20240829-202402.csv"

    df_dict = prepare_dataframe(summarize_data)

    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Precision"] == "float32"]

    df_dict["mv2"] = df_dict["original"].loc[df_dict['original']['Model version'] == "v2"]
    df_dict["mv2"]["Model Median per Image"] = df_dict['mv2']["Model Median"] / df_dict['mv2']["Batch size"]
    df_dict['mv2']["Batch size"] = pd.Categorical(df_dict['mv2']["Batch size"])

    plot_df = df_dict["mv2"]

    plot = p9.ggplot(plot_df, p9.aes(x='Net Model', y='Model Median per Image', fill='Batch size'))
    plot += p9.geom_col(stat='identity', position='dodge')
    prepare_plot(plot, f'mv2_s24_CPU5_bs_{time.strftime("%Y%m%d-%H%M%S")}', 'Model', 'Czas inferencji [ms]', angle=90)

    # ----before----
    # df_dict["createdAtNotNull"] = df_dict["default_models"].loc[df_dict["default_models"]["CreatedAt"].notnull()]
    # df_dict["s24_notnull"] = df_dict["createdAtNotNull"].loc[
    #     df_dict["createdAtNotNull"]["Phone"] == 'Samsung Galaxy S24']
    # df_dict["s24_notnull_CPU"] = df_dict['s24_notnull'].loc[df_dict['s24_notnull']['Delegate'] == 'CPU']
    # df_dict["s24_notnull_CPU_20240712"] = df_dict['s24_notnull_CPU'].loc[
    #     df_dict['s24_notnull_CPU']['CreatedAt'].str.contains("2024-07-12")]
    # df_dict["mv2"] = df_dict["s24_notnull_CPU_20240712"].loc[
    #     df_dict['s24_notnull_CPU_20240712']['Model version'].str.contains("v2")]
    # df_dict["mv2"]["Model Median per Image"] = df_dict['mv2']["Model Median"] / df_dict['mv2']["Batch size"]
    # df_dict['mv2']["Model Median per Image"] = df_dict['mv2']["Model Median per Image"].astype(int)
    # df_dict['mv2']["Batch size"] = pd.Categorical(df_dict['mv2']["Batch size"])
    #
    # plot_df = df_dict["mv2"]
    #
    # plot = p9.ggplot(plot_df, p9.aes(x='Net Model', y='Model Median per Image', fill='Batch size'))
    # plot += p9.geom_col(stat='identity', position='dodge')
    # prepare_plot(plot, "mv2_s24_CPU_bs", 'Model', 'Czas inferencji [ns]', angle=90)


def compare_bs_mv2_1_0_224(phone="OnePlus 9 Pro"):
    # lista plikow dla s24
    # files = ["Results_summarized_20240712-164920_bs_CPU1.csv", "Results_summarized_20240829-202402.csv",
    #          "Results_summarized_20240829-184554_changed_mv1.csv"]

    files = ['Results_summarized_20240907-130256.csv']
    li = []

    for filename in files:
        df_tmp = pd.read_csv(os.path.join(CSV_PATH, filename), index_col=None, header=0)
        li.append(df_tmp)

    df = pd.concat(li, axis=0, ignore_index=True)

    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds

    # rename mobilenet_96 to mobilenet_096
    df = df.replace("_96", "_096", regex=True)
    # rename mobilenet_v3_1.0_224_uint8 to mobilenet_v3_1.0_224_large_uint8
    df = df.replace("mobilenet_v3_1.0_224_uint8", "mobilenet_v3_1.0_224_large_uint8", regex=True)

    split_dataframe(df)

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df
    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Phone"] == phone]
    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Precision"] == "float32"]
    df_dict['original'] = df_dict['original'].loc[df_dict['original']["Net Model"].str.contains("1.0_224")]

    df_dict["mv2"] = df_dict["original"].loc[df_dict['original']['Model version'] == "v2"]
    df_dict["mv2"]["Model Median per Image"] = df_dict['mv2']["Model Median"] / df_dict['mv2']["Batch size"]
    # df_dict['mv1']["Model Median per Image"] = df_dict['mv1']["Model Median per Image"].astype(int)
    df_dict['mv2']["Batch size"] = pd.Categorical(df_dict['mv2']["Batch size"])

    plot_df = df_dict["mv2"]

    plot = p9.ggplot(plot_df, p9.aes(x='Delegate', y='Model Median per Image', fill='Batch size'))
    plot += p9.geom_col(stat='identity', position='dodge')
    prepare_plot(plot, f'6_3_2_{phone}_bs_mv2_1_0_224_{time.strftime("%Y%m%d-%H%M%S")}', 'Delegat',
                 'Czas inferencji [ms]', angle=0)


def compare_bs_mv3():
    # summarize_data = "Results_summarized_20240829-184554.csv"
    # summarize_data = "Results_summarized_20240712-164920_bs_CPU1_add_other_rho.csv"
    summarize_data = "Results_summarized_20240829-202402.csv"

    df_dict = prepare_dataframe(summarize_data)

    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Precision"] == "float32"]

    df_dict["mv3"] = df_dict["original"].loc[df_dict['original']['Model version'].str.contains("v3")]
    df_dict["mv3"]["Model Median per Image"] = df_dict['mv3']["Model Median"] / df_dict['mv3']["Batch size"]
    df_dict['mv3']["Batch size"] = pd.Categorical(df_dict['mv3']["Batch size"])

    plot_df = df_dict["mv3"]

    plot = p9.ggplot(plot_df, p9.aes(x='Net Model', y='Model Median per Image', fill='Batch size'))
    plot += p9.geom_col(stat='identity', position='dodge')
    prepare_plot(plot, f'mv3_s24_CPU4_bs_{time.strftime("%Y%m%d-%H%M%S")}', 'Model', 'Czas inferencji [ms]', angle=90)

    # ---- before ----
    # df_dict["createdAtNotNull"] = df_dict["default_models"].loc[df_dict["default_models"]["CreatedAt"].notnull()]
    # df_dict["s24_notnull"] = df_dict["createdAtNotNull"].loc[
    #     df_dict["createdAtNotNull"]["Phone"] == 'Samsung Galaxy S24']
    # df_dict["s24_notnull_CPU"] = df_dict['s24_notnull'].loc[df_dict['s24_notnull']['Delegate'] == 'CPU']
    # df_dict["mv3"] = df_dict["s24_notnull_CPU"].loc[df_dict['s24_notnull_CPU']['Model version'].str.contains("v3")]
    # df_dict["mv3"]["Model Median per Image"] = df_dict['mv3']["Model Median"] / df_dict['mv3']["Batch size"]
    # df_dict['mv3']["Model Median per Image"] = df_dict['mv3']["Model Median per Image"].astype(int)
    # df_dict['mv3']["Batch size"] = pd.Categorical(df_dict['mv3']["Batch size"])
    #
    # plot_df = df_dict["mv3"]
    #
    # plot = p9.ggplot(plot_df, p9.aes(x='Net Model', y='Model Median per Image', fill='Batch size'))
    # plot += p9.geom_col(stat='identity', position='dodge')
    # prepare_plot(plot, "mv3_s24_CPU_bs", 'Model', 'Czas inferencji [ns]', angle=90)


def compare_bs_mv3_1_0_224(phone='Motorola Razr 50 Ultra'):
    # lista plikow dla s24
    # files = ["Results_summarized_20240712-164920_bs_CPU1.csv", "Results_summarized_20240829-202402.csv",
    #          "Results_summarized_20240829-184554_changed_mv1.csv"]

    files = ['Results_summarized_20240907-130256.csv']
    li = []

    for filename in files:
        df_tmp = pd.read_csv(os.path.join(CSV_PATH, filename), index_col=None, header=0)
        li.append(df_tmp)

    df = pd.concat(li, axis=0, ignore_index=True)

    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds

    # rename mobilenet_96 to mobilenet_096
    df = df.replace("_96", "_096", regex=True)
    # rename mobilenet_v3_1.0_224_uint8 to mobilenet_v3_1.0_224_large_uint8
    df = df.replace("mobilenet_v3_1.0_224_uint8", "mobilenet_v3_1.0_224_large_uint8", regex=True)

    split_dataframe(df)

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df
    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Phone"] == phone]
    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Precision"] == "float32"]
    df_dict['original'] = df_dict['original'].loc[df_dict['original']["Net Model"].str.contains("1.0_224")]

    df_dict["mv3"] = df_dict["original"].loc[(df_dict['original']['Model version'].str.contains("v3_large")) & (
        ~df_dict['original']['Model version'].str.contains("minimalistic"))]
    df_dict["mv3"]["Model Median per Image"] = df_dict['mv3']["Model Median"] / df_dict['mv3']["Batch size"]
    df_dict['mv3']["Batch size"] = pd.Categorical(df_dict['mv3']["Batch size"])

    plot_df = df_dict["mv3"]

    plot = p9.ggplot(plot_df, p9.aes(x='Delegate', y='Model Median per Image', fill='Batch size'))
    plot += p9.geom_col(stat='identity', position='dodge')
    prepare_plot(plot, f'6_3_2_{phone}_bs_mv3_1_0_224_{time.strftime("%Y%m%d-%H%M%S")}', 'Delegat',
                 'Czas inferencji [ms]', angle=0)


# def compare_quant(df_dict):
#     model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['s24_all'], p9.aes(x='Model Median', y='Model mAP'))
#     model_delay_s24_float_vs_uint8 += p9.geom_point(p9.aes(fill='Precision', shape="Delegate", color="Model version"))
#     model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
#     prepare_plot(model_delay_s24_float_vs_uint8, 'mv1_s24_s24_float_vs_uint8', 'Czas inferencji [ns]',
#                  'Top1 trafność [%]',
#                  angle=90)


def compare_quant_old(df_dict):
    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['s24_all'], p9.aes(x='Model Median', y='Model mAP'))
    model_delay_s24_float_vs_uint8 += p9.geom_point(
        p9.aes(fill='Precision', shape="Delegate", color="Model version", size="Precision"))
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(x='Model Median', y='Model mAP', label='label'))
    prepare_plot(model_delay_s24_float_vs_uint8, 'mv1_s24_s24_float_vs_uint8', 'Czas inferencji [ns]',
                 'Top1 trafność [%]',
                 angle=90)


def compare_quant(df_dict):
    df_dict["createdAtNotNull"] = df_dict["original"].loc[df_dict["original"]["CreatedAt"].notnull()]
    df_dict["s24_notnull"] = df_dict["createdAtNotNull"].loc[
        df_dict["createdAtNotNull"]["Phone"] == 'Samsung Galaxy S24']
    df_dict["s24_notnull_CPU"] = df_dict['s24_notnull'].loc[df_dict['s24_notnull']['Delegate'] == 'CPU']
    df_dict["s24_notnull_CPU_202408"] = df_dict['s24_notnull_CPU'].loc[
        df_dict['s24_notnull_CPU']['CreatedAt'].str.contains("2024-08")]
    df_dict["s24_notnull_CPU_202408_mv1"] = df_dict["s24_notnull_CPU_202408"].loc[
        df_dict['s24_notnull_CPU_202408']['Model version'] == "v1"]
    df_dict["s24_notnull_CPU_202408_mv2"] = df_dict["s24_notnull_CPU_202408"].loc[
        df_dict['s24_notnull_CPU_202408']['Model version'] == "v2"]
    df_dict["s24_notnull_CPU_202408_mv3"] = df_dict["s24_notnull_CPU_202408"].loc[
        df_dict['s24_notnull_CPU_202408']['Model version'].str.contains("v3")]

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['s24_notnull_CPU_202408_mv1'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black', adjust_text={
        'expand_points': (1.5, 1.5),
        'arrowprops': {
            'arrowstyle': '-'
        }
    })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv1_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ns]',
                 'Top1 trafność [%]',
                 angle=90)

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['s24_notnull_CPU_202408_mv2'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black', adjust_text={
        'expand_points': (1.5, 1.5),
        'arrowprops': {
            'arrowstyle': '-'
        }
    })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv2_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ns]',
                 'Top1 trafność [%]',
                 angle=90)

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['s24_notnull_CPU_202408_mv3'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black', adjust_text={
        'expand_points': (1.5, 1.5),
        'arrowprops': {
            'arrowstyle': '-'
        }
    })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv3_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ns]',
                 'Top1 trafność [%]',
                 angle=90)


def s24_CPU4_compare_models():
    summarize_data = "Results_summarized_20240816-164452.csv"
    df_dict = prepare_dataframe(summarize_data)
    df_dict["s24_1.0_224_float32"] = df_dict["original"].loc[(df_dict["original"]["alpha"] == "1.0") &
                                                             (df_dict["original"]["rho"] == "224") &
                                                             (df_dict["original"]["Precision"] == "float32")]

    plot = p9.ggplot(df_dict["s24_1.0_224_float32"], p9.aes(x='Net Model', y='Model Median'))
    plot += p9.geom_col(stat='identity', position='dodge')
    prepare_plot(plot, f's24_CPU4_compare_models_{time.strftime("%Y%m%d-%H%M%S")}', 'Model', 'Czas inferencji [ms]',
                 angle=90)


def s24_compare_delegates():
    files = ["Results_summarized_20240816-164452.csv",  # CPU4
             "Results_summarized_20240816-174401.csv",  # CPU1
             "Results_summarized_20240826-200110.csv",  # CPU5
             "Results_summarized_20240826-202547.csv",  # CPU6
             "Results_summarized_20240816-180852.csv",  # GPU
             "Results_summarized_20240816-184333.csv"]  # NNAPI
    # CSV_PATH = "D:/Dokumenty/Mgr - CNNs/Moje_wyniki_testów/"
    # files = ["Results_added_CPU5_CPU6.csv"]
    # files = ["Results_summarized_20240910-203945.csv"]
    li = []

    for filename in files:
        df_tmp = pd.read_csv(os.path.join(CSV_PATH, filename), index_col=None, header=0)
        li.append(df_tmp)

    df = pd.concat(li, axis=0, ignore_index=True)

    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds

    # rename mobilenet_96 to mobilenet_096
    df = df.replace("_96", "_096", regex=True)
    # rename mobilenet_v3_1.0_224_uint8 to mobilenet_v3_1.0_224_large_uint8
    df = df.replace("mobilenet_v3_1.0_224_uint8", "mobilenet_v3_1.0_224_large_uint8", regex=True)

    split_dataframe(df)

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df
    df_dict['original'] = df_dict["original"].loc[df_dict["original"]["Phone"] == "Samsung Galaxy S24"]
    df_dict["s24_1.0_224_float32"] = df_dict["original"].loc[(df_dict["original"]["alpha"] == "1.0") &
                                                             (df_dict["original"]["rho"] == "224") &
                                                             (df_dict["original"]["Precision"] == "float32") &
                                                             (df_dict["original"]["Batch size"] == 1)]

    df_dict["mv1"] = df_dict["s24_1.0_224_float32"].loc[
        df_dict["s24_1.0_224_float32"]['Model version'].str.contains("v1")]
    df_dict["mv2"] = df_dict["s24_1.0_224_float32"].loc[
        df_dict["s24_1.0_224_float32"]['Model version'].str.contains("v2")]
    df_dict["mv3"] = df_dict["s24_1.0_224_float32"].loc[
        df_dict["s24_1.0_224_float32"]['Model version'].str.contains("v3")]

    # CPU vs GPU vs NNAPI s24 mv1
    model_delay_s24_CPUvsGPUvsNNAPI_v1 = p9.ggplot(df_dict['mv1'],
                                                   p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsNNAPI_v1 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsNNAPI_v1 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v1, f'mv1_s24_GPUvsCPUvsNNAPI_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ms]',
                 angle=90)

    # CPU vs GPU vs NNAPI s24 mv2
    model_delay_s24_CPUvsGPUvsNNAPI_v2 = p9.ggplot(df_dict['mv2'],
                                                   p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsNNAPI_v2 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsNNAPI_v2 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v2, f'mv2_s24_GPUvsCPUvsNNAPI_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ms]',
                 angle=90)

    # CPU vs GPU vs NNAPI s24 mv3
    model_delay_s24_CPUvsGPUvsNNAPI_v3 = p9.ggplot(df_dict['mv3'],
                                                   p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsNNAPI_v3 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsNNAPI_v3 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_v3, f'mv3_s24_GPUvsCPUvsNNAPI_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ms]',
                 angle=90)

    model_delay_s24_CPUvsGPUvsNNAPI_all = p9.ggplot(df_dict['s24_1.0_224_float32'],
                                                    p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsNNAPI_all += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsNNAPI_all += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsNNAPI_all, f'class_delegates_all_s24_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ms]',
                 angle=90)


#
def compare_quant_CPU4():
    summarize_data = "Results_summarized_20240816-164452.csv"
    df_dict = prepare_dataframe(summarize_data)

    df_dict["mv1"] = df_dict["original"].loc[
        df_dict['original']['Model version'] == "v1"]
    df_dict["mv2"] = df_dict["original"].loc[
        df_dict['original']['Model version'] == "v2"]
    df_dict["mv3"] = df_dict["original"].loc[
        df_dict['original']['Model version'].str.contains("v3")]

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['mv1'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black', adjust_text={
        'expand_points': (1.5, 1.5),
        'arrowprops': {
            'arrowstyle': '-'
        }
    })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv1_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ms]',
                 'Top1 trafność [%]',
                 angle=90)

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['mv2'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black', adjust_text={
        'expand_points': (1.5, 1.5),
        'arrowprops': {
            'arrowstyle': '-'
        }
    })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv2_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ms]',
                 'Top1 trafność [%]',
                 angle=90)

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['mv3'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black', adjust_text={
        'expand_points': (1.5, 1.5),
        'arrowprops': {
            'arrowstyle': '-'
        }
    })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv3_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ms]',
                 'Top1 trafność [%]',
                 angle=90)


def compare_quant_CPU1():
    summarize_data = "Results_summarized_20240816-174401.csv"
    df_dict = prepare_dataframe(summarize_data)

    df_dict["mv1"] = df_dict["original"].loc[
        df_dict['original']['Model version'] == "v1"]
    df_dict["mv2"] = df_dict["original"].loc[
        df_dict['original']['Model version'] == "v2"]
    df_dict["mv3"] = df_dict["original"].loc[
        df_dict['original']['Model version'].str.contains("v3")]

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['mv1'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black',
                                                   adjust_text={
                                                       'expand_points': (1.5, 1.5),
                                                       'arrowprops': {
                                                           'arrowstyle': '-'
                                                       }
                                                   })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv1_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ms]',
                 'Top1 trafność [%]',
                 angle=90)

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['mv2'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black',
                                                   adjust_text={
                                                       'expand_points': (1.5, 1.5),
                                                       'arrowprops': {
                                                           'arrowstyle': '-'
                                                       }
                                                   })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv2_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ms]',
                 'Top1 trafność [%]',
                 angle=90)

    model_delay_s24_float_vs_uint8 = p9.ggplot(df_dict['mv3'],
                                               p9.aes(x='Model Median', y='Model mAP', color='Precision'))
    model_delay_s24_float_vs_uint8 += p9.geom_point()
    model_delay_s24_float_vs_uint8 += p9.scale_fill_discrete(name=" ")
    model_delay_s24_float_vs_uint8 += p9.geom_text(mapping=p9.aes(label='label'), size=7, color='black',
                                                   adjust_text={
                                                       'expand_points': (1.5, 1.5),
                                                       'arrowprops': {
                                                           'arrowstyle': '-'
                                                       }
                                                   })
    prepare_plot(model_delay_s24_float_vs_uint8, f'mv3_s24_float_vs_uint8_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Czas inferencji [ms]',
                 'Top1 trafność [%]',
                 angle=90)


def compare_delegates_detection(summarize_data, phone='Samsung Galaxy S24'):
    CSV_PATH = "D:/Dokumenty/Mgr - CNNs/Moje_wyniki_testów/"

    df = pd.read_csv(os.path.join(CSV_PATH, summarize_data))
    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df
    df_dict['original'] = df_dict['original'].loc[df_dict["original"]["Phone"] == phone]
    df_dict['original'] = df_dict['original'].loc[df_dict['original']['CreatedAt'].str.contains("2024-09-14")]
    df_dict['mobilenets'] = df_dict['original'].loc[~df["Net Model"].str.contains("V5s")]
    df_dict['yolo'] = df_dict['original'].loc[~df["Net Model"].str.contains("Mobile")]

    # CPU vs GPU vs CPU4 s24
    model_delay_s24_CPUvsGPUvsCPU4 = p9.ggplot(df_dict['original'],
                                               p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsCPU4 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsCPU4 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsCPU4, f'7_1_1_{phone}_detection_delegates_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ms]',
                 angle=0)

    # CPU vs GPU vs CPU4 s24 mobilenet
    model_delay_s24_CPUvsGPUvsCPU4 = p9.ggplot(df_dict['mobilenets'],
                                               p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsCPU4 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsCPU4 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsCPU4,
                 f'7_1_1_{phone}_detection_delegates_mobilenets_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ms]',
                 angle=0)

    # CPU vs GPU vs CPU4 s24 yolo
    model_delay_s24_CPUvsGPUvsCPU4 = p9.ggplot(df_dict['yolo'],
                                               p9.aes(x='Net Model', y='Model Median'))
    model_delay_s24_CPUvsGPUvsCPU4 += p9.geom_col(p9.aes(fill='Delegate'), stat='identity', position='dodge')
    model_delay_s24_CPUvsGPUvsCPU4 += p9.scale_fill_discrete(name=" ")
    prepare_plot(model_delay_s24_CPUvsGPUvsCPU4,
                 f'7_1_1_{phone}_detection_delegates_yolo_{time.strftime("%Y%m%d-%H%M%S")}',
                 'Model',
                 'Czas inferencji [ms]',
                 angle=0)


def remove_dominated(df):
    df["dominated"] = 0
    row_count = len(df)

    for i in range(row_count):
        for j in range(row_count):
            if i != j and df.loc[i, "Model version"] == df.loc[j, "Model version"]:
                if df.loc[i, "Model Median"] > df.loc[j, "Model Median"] and df.loc[i, "Model mAP"] <= df.loc[
                    j, "Model mAP"]:
                    df.loc[i, "dominated"] = 1
                elif df.loc[i, "Model Median"] < df.loc[j, "Model Median"] and df.loc[i, "Model mAP"] >= df.loc[
                    j, "Model mAP"]:
                    df.loc[j, "dominated"] = 1


def find_pareto_optimal(df):
    pareto_optimal = []

    for i, row in df.iterrows():
        dominated = False
        for j, other_row in df.iterrows():
            if i != j and df.loc[i, "Model version"] == df.loc[j, "Model version"]:
                if (other_row['Model Median'] <= row['Model Median'] and
                        other_row['Model mAP'] >= row['Model mAP'] and
                        (other_row['Model Median'] < row['Model Median'] or
                         other_row['Model mAP'] > row['Model mAP'])):
                    dominated = True
                    break
        if not dominated:
            pareto_optimal.append(row)

    return pd.DataFrame(pareto_optimal)


def classification_compare_all_parameters(phone='Samsung Galaxy S24'):
    # poprzednie pliki osobno dla scenariuszy S24 na CPU4, CPU1, GPU, NNAPI
    # files = ["Results_summarized_20240816-164452.csv", "Results_summarized_20240816-174401.csv",
    #          "Results_summarized_20240816-180852.csv", "Results_summarized_20240816-184333.csv"]

    # poprzedni plik z oneplus i motorola bez s23 i bez xiaomi
    # files = ['Results_summarized_20240907-130256.csv']
    files = ['Results_summarized_20240910-203945.csv']
    li = []

    for filename in files:
        df_tmp = pd.read_csv(os.path.join(CSV_PATH, filename), index_col=None, header=0)
        li.append(df_tmp)

    df = pd.concat(li, axis=0, ignore_index=True)

    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds

    # rename mobilenet_96 to mobilenet_096
    df = df.replace("_96", "_096", regex=True)
    # rename mobilenet_v3_1.0_224_uint8 to mobilenet_v3_1.0_224_large_uint8
    df = df.replace("mobilenet_v3_1.0_224_uint8", "mobilenet_v3_1.0_224_large_uint8", regex=True)

    split_dataframe(df)

    df = df[df['Phone'] == phone]
    df = df[df['Batch size'] == 1]

    # remove_dominated(df)
    df = find_pareto_optimal(df)

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df
    # df_dict[phone] = df_dict['original'].loc[(df_dict["original"]["Phone"] == phone)]

    # df_dict["without_dominated"] = df_dict[phone].loc[(df_dict[phone]["dominated"] == 0)]
    df_dict["original"]['Version-Precision'] = df_dict["original"]["Model version"] + "_" + \
                                               df_dict["original"]["Precision"]

    x = 1
    y = 1.5
    r = 0.5
    t = 1
    plot = p9.ggplot(df_dict['original'],
                     p9.aes(x='Model Median',
                            y='Model mAP',
                            color='Model version',
                            shape='Delegate'
                            ))
    plot += p9.geom_point(p9.aes(fill='Version-Precision'), stroke=1.5, size=5)
    plot += p9.geom_line(p9.aes(group='Model version'), linetype='dashed')
    plot += p9.scale_shape_manual(values={'CPU': 's', 'CPU4': 'D', 'GPU': 'o', 'NNAPI': 'X'})
    plot += p9.scale_color_manual(
        values={"v1": "#db5f57", "v2": "#d4db57", "v3_large": "#5e57db", "v3_small": "#dc57d4"})
    plot += p9.scale_fill_manual(
        values={"v1_uint8": "white", "v2_uint8": "white", "v3_large_uint8": "white", "v3_small_uint8": "white",
                "v1_float32": "#db5f57", "v2_float32": "#d4db57", "v3_large_float32": "#5e57db",
                "v3_small_float32": "#dc57d4"})
    # plot += p9.scale_fill_discrete(name=" ")
    plot += p9.geom_text(mapping=p9.aes(label='label'), size=8, ha='center', va='center', color='black',
                         adjust_text={
                             'expand': (x, y, r, t),
                             'arrowprops': {
                                 'arrowstyle': '-'
                             }
                         })
    plot += p9.labs(color="Wersja modelu", shape='Delegat', fill='Precyzja')
    p9.scale_fill_manual(values={'float32': '#000000', 'uint8': 'white'},
                         guide=p9.guide_legend(reverse=True))
    prepare_plot(plot, f'6_3_3_{phone}_all_{time.strftime("%Y%m%d-%H%M%S")}_{x}_{y}_{r}_{t}',
                 'Czas inferencji [ms]',
                 'Top1 trafność [%]',
                 angle=0)
    # plot = (
    #         ggplot(df_dict["without_dominated"], aes(x='Model Median', y='Model mAP'))
    #         + geom_point(aes(
    #                         stroke=1.5,
    #                         shape='Delegate',
    #                         size='Precision',  # tylko po to, aby precision była w legendzie
    #                         color='Model version',
    #                         fill=np.where(df_dict["without_dominated"]['Precision'] == "float32", df_dict["without_dominated"]['Model version'], np.nan)
    #                     ))
    #         + scale_shape_manual(values=[24, 21, 23])
    #         + scale_fill_discrete(name=" ")
    #         + scale_color_manual(name="model", values=["red", "yellow", "#000000", "blue", "#80FFFF"],
    #                              aesthetics=["colour", "fill"])
    #         + guides(colour=guide_legend(override_aes={'shape': 15}))  # użycie kształtu, który nie jest zarezerwowany
    #         + guides(
    #             shape=guide_legend(override_aes={'fill': 'black'}))  # użycie koloru wypełnienia, który nie jest zarezerwowany
    #         + scale_size_manual(name="precision", values=[2, 2])  # chcemy, aby float32 i uint8 miały ten sam rozmiar
    #         + guides(size=guide_legend(override_aes={'shape': [15, 0]}))
    #         + xlab("Inference latency [ms]")
    #         + ylab("Top1 accuracy [%]")
    #         + geom_text(mapping=p9.aes(label='label'), size=7, color='black',
    #                                                adjust_text={
    #                                                    'expand_points': (1.5, 1.5),
    #                                                    'arrowprops': {
    #                                                        'arrowstyle': '-'
    #                                                    }
    #                                                })
    #         + theme_bw()
    #         + theme(plot_title=element_text(size=22), axis_title=element_text(size=14), axis_text=element_text(size=13))
    # )
    # save_gg_plot(plot, f'test_{time.strftime("%Y%m%d-%H%M%S")}')
    # prepare_plot(plot, f'test_{time.strftime("%Y%m%d-%H%M%S")}',
    #              'Czas inferencji [ms]',
    #              'Top1 trafność [%]',
    #              angle=90)


def detection_compare_accuracy_latency(phone='Motorola Razr 50 Ultra'):
    CSV_PATH = "D:/Dokumenty/Mgr - CNNs/Moje_wyniki_testów"
    filename = "Results_summarized_20240917-181701.csv"
    df = pd.read_csv(os.path.join(CSV_PATH, filename), index_col=None, header=0)

    df['Model Median'] = df['Model Median'].astype(np.int64) / int(1e6)  # zmiana nanoseconds to miliseconds

    # rename mobilenet_96 to mobilenet_096
    df = df.replace("_96", "_096", regex=True)
    # rename mobilenet_v3_1.0_224_uint8 to mobilenet_v3_1.0_224_large_uint8
    df = df.replace("mobilenet_v3_1.0_224_uint8", "mobilenet_v3_1.0_224_large_uint8", regex=True)

    # split_dataframe(df)

    df = df[df['Phone'] == phone]
    # df = df[df['CreatedAt'].str.contains('2024-09-14')]
    df = df[~df['Net Model'].str.contains('Yolo')]

    df_dict: Dict[str, pd.DataFrame] = {}
    df_dict['original'] = df

    plot = (ggplot(df_dict['original'], aes(x='Model Median', y='Model mAP', fill='Net Model', shape='Delegate'))
            + geom_point(size=5, stroke=1, color='black')
            # + geom_text(aes(label='Net Model'), nudge_y=-1, size=13, color='black')
            + scale_shape_manual(values={'CPU1': 's', 'CPU4': 'D', 'GPU': 'o', 'CPU2': 'v', 'CPU5': 'p', 'CPU6': '*'})
            + scale_fill_manual(
                values={'MobileNetV1': '#db5f57', 'MobileNetV2': '#d4db57', 'MobileNetV3-large': '#5e57db',
                        'MobileNetV3-small': '#dc57d4', 'YOLOv5s': '#515412'})
            # + scale_x_log10()
            + labs(x="Czas inferencji [ms]", y="mAP [%]", shape='Delegat', fill="Wersja modelu")
            + theme(figure_size=(8, 6))
            )

    save_gg_plot(plot, f'7_3_1_{phone}_all_{time.strftime("%Y%m%d-%H%M%S")}')


if __name__ == '__main__':
    # summarize_data = "Results_summarized_20240810-114534.csv"
    # summarize_data = "Results_added_CPU5_CPU6.csv.csv"
    # df_dict = prepare_dataframe()
    #
    # compare_delegates(df_dict)

    # compare_quant(df_dict)

    # compare_bs_mv1()
    # compare_bs_mv1_1_0_224()

    # compare_bs_mv2()
    # compare_bs_mv2_1_0_224()

    # compare_bs_mv3()
    # compare_bs_mv3_1_0_224()

    # compare_quant(df_dict)

    # s24_CPU4_compare_models()

    # s24_compare_delegates()

    # compare_quant_CPU4()

    # compare_quant_CPU1()

    classification_compare_all_parameters('Samsung Galaxy S24')

    # --- object detection ---

    # compare_delegates_detection("Results_summarized_20240917-181701.csv")

    # detection_compare_accuracy_latency()
