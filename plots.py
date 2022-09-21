import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

run_dirs = [

    # Windows
    # 'RUNS/VALID/PCBA/PCBA-FNDS-V1',
    # 'RUNS/VALID/ZINC/ZINC-FNDS-V1',
    # 'RUNS/VALID/ZINC/ZINC-SOPR-V1',
    # 'RUNS/2022-09-06@16-45-59-manshari-desktop-ZINC-FNDS-PSC',
    # 'RUNS/2022-09-06@20-37-52-manshari-desktop-ZINC-SOPR-PSC',
    # 'RUNS/2022-09-07@08-02-22-manshari-desktop-PCBA-FNDS-PSC',
    # 'RUNS/2022-09-06@10-50-26-manshari-desktop-PCBA-SOPR-PSC',
    # 'RUNS/2022-09-07@13-16-52-manshari-desktop-PCBA-SOPR-PSC',
    # 'RUNS/2022-09-07@18-19-41-manshari-desktop-ZINC-FNDS-PSC',
    # 'RUNS/2022-09-07@21-30-19-manshari-desktop-ZINC-SOPR-PSC',
    'RUNS/2022-09-08@00-34-43-manshari-desktop-PCBA-FNDS-PSC',
    'RUNS/2022-09-08@05-46-33-manshari-desktop-PCBA-SOPR-PSC',
    # 'RUNS/2022-09-08@10-46-36-manshari-desktop-ZINC-FNDS-PSC',
    # 'RUNS/2022-09-08@13-55-31-manshari-desktop-ZINC-SOPR-PSC',
    'RUNS/2022-09-08@16-59-11-manshari-desktop-PCBA-FNDS-PSC',
    'RUNS/2022-09-08@22-12-05-manshari-desktop-PCBA-SOPR-PSC',
    # 'RUNS/2022-09-09@03-13-41-manshari-desktop-ZINC-FNDS-PSC',
    # 'RUNS/2022-09-09@06-21-29-manshari-desktop-ZINC-SOPR-PSC',

    # Linux
    # PROJ_DIR / 'RUNS/2022-08-17@10_10_40-manshari-desktop-ZINC',
    # PROJ_DIR / 'RUNS/2022-08-16@08_02_19-manshari-desktop-PCBA',
    # PROJ_DIR / 'RUNS/2022-08-18@09_54_34-manshari-desktop-ZINCMOSES',
]

# penguins = sns.load_dataset("penguins")
# print(penguins.head(5))

df = None
bank = ['FNDS', 'SOPR'] * 3
count = 0

sns.set(style='darkgrid')
for idx, run_dir in enumerate(run_dirs):
    finalGenStr = run_dir + '/results/samples_del/new_pop_final.csv'
    finalGenData = pd.read_csv(
        finalGenStr,
        index_col=0,
        skip_blank_lines=True
    )
    # properties = ['qed', 'SAS', 'logP', 'rank']
    properties = ['qed', 'SAS', 'logP']
    # print("properties", properties)

    original_data_length = len(finalGenData)
    print("Number of Rows:", original_data_length)
    finalGenData.dropna(subset=properties, inplace=True)
    finalGenData.reset_index(inplace=True, drop=True)
    post_dropna_data_length = len(finalGenData)
    print("Number of Non-Null Samples:", post_dropna_data_length)
    rank_type = [f"{bank[idx]}{idx}"] * post_dropna_data_length
    rt = pd.DataFrame({'rank_type': rank_type})
    # print(rt.head())
    # print(len(rt))
    dataset = pd.concat([rt, finalGenData], axis=1)
    # print(dataset.head())
    # print(len(dataset))
    if df is None:
        df = dataset
    else:
        df = pd.concat([df, dataset]).reset_index(drop=True)

sns.displot(df, x='qed', hue='rank_type', kind='kde', legend=True)
sns.displot(df, x='SAS', hue='rank_type', kind='kde', legend=True)
sns.displot(df, x='logP', hue='rank_type', kind='kde', legend=True)

#
# # plotting both distibutions on the same figure
# fig = sns.kdeplot(df['sepal_width'], shade=True, color="r")
# fig = sns.kdeplot(df['sepal_length'], shade=True, color="b")

plt.legend()
plt.show()
